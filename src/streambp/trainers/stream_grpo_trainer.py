import torch
from trl import GRPOTrainer
from trl.trainer.utils import selective_log_softmax
from trl.trainer.grpo_trainer import nanmin, nanmax
from ..stream_model import StreamModel
from typing import Any, Callable, Optional, Union

MAX_PAD_RATIO = 0.2  # Maximum ratio of padding tokens in a sequence
torch.manual_seed(0)
torch.cuda.manual_seed(0)

class OriginalGRPOTrainer(GRPOTrainer):
    def __init__(
            self, *args, **kwargs
        ) -> None:
        self.vocab_size = kwargs["model"].lm_head.out_features
        self.max_completion_length = kwargs.pop("max_completion_length") # NOTE: for generating dummy data only; to delete after testing
        super().__init__(**kwargs)
        
    def _prepare_inputs(self, inputs):
        # randomly generate some data for memory profiling
        keys = inputs[0].keys()
        batch_size = len(inputs)
        device = self.model.device

        batched_inputs = {}
        for key in keys:
            batched_inputs[key] = torch.vstack([torch.tensor(inputs[i][key]) for i in range(len(inputs))]).to(device)

        completion_ids = torch.randint(0, self.vocab_size, (batch_size, self.max_completion_length))
        completion_mask = torch.ones_like(completion_ids)
        for mask in completion_mask:
            mask[-torch.randint(1, int(self.max_completion_length*MAX_PAD_RATIO), (1,)):] = 0

        advantage = torch.randn(batch_size)
        old_per_token_logps = torch.randn(batch_size, self.max_completion_length)
        ref_per_token_logps = torch.randn(batch_size, self.max_completion_length)

        batched_inputs["completion_ids"] = completion_ids.to(device)
        batched_inputs["completion_mask"] = completion_mask.to(device)
        batched_inputs["advantages"] = advantage.to(device)
        batched_inputs["old_per_token_logps"] = old_per_token_logps.to(device)
        batched_inputs["ref_per_token_logps"] = ref_per_token_logps.to(device)

        return batched_inputs

class StreamGRPOTrainer(GRPOTrainer):
    def __init__(
            self, *args, **kwargs
        ) -> None:
        # assert isinstance(kwargs["model"], StreamModel), "model must be a StreamModel"
        self.chunk_size = kwargs["model"].logits_chunk_size
        self.vocab_size = kwargs["model"].model.lm_head.out_features
        self.max_completion_length = kwargs.pop("max_completion_length") # NOTE: for generating dummy data only; to delete after testing
        super().__init__(**kwargs)
        self.accelerator.backward = lambda loss: None # backward is fused with forward, no need to call accelerator.backward
        
    def _get_chunk_logits(self, model, hidden_states, chunk_slice):
        logits = model.lm_head(hidden_states[:, chunk_slice, :])

        return logits
    
    def _prepare_inputs(self, inputs):
        # randomly generate some data for memory profiling
        keys = inputs[0].keys()
        batch_size = len(inputs)
        device = self.model.device

        batched_inputs = {}
        for key in keys:
            batched_inputs[key] = torch.vstack([torch.tensor(inputs[i][key]) for i in range(len(inputs))]).to(device)

        completion_ids = torch.randint(0, self.vocab_size, (batch_size, self.max_completion_length))
        completion_mask = torch.ones_like(completion_ids)
        for mask in completion_mask:
            mask[-torch.randint(1, int(self.max_completion_length*MAX_PAD_RATIO), (1,)):] = 0

        advantage = torch.randn(batch_size)
        old_per_token_logps = torch.randn(batch_size, self.max_completion_length)
        ref_per_token_logps = torch.randn(batch_size, self.max_completion_length)

        batched_inputs["completion_ids"] = completion_ids.to(device)
        batched_inputs["completion_mask"] = completion_mask.to(device)
        batched_inputs["advantages"] = advantage.to(device)
        batched_inputs["old_per_token_logps"] = old_per_token_logps.to(device)
        batched_inputs["ref_per_token_logps"] = ref_per_token_logps.to(device)

        return batched_inputs

    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep, batch_size=None) -> torch.Tensor:
        batch_size = batch_size or input_ids.size(0)  # Chunk inputs into smaller batches to reduce memory peak
        all_logps = []
        for i in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[i : i + batch_size]
            attention_mask_batch = attention_mask[i : i + batch_size]

            # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
            logits = model(
                input_ids=input_ids_batch, attention_mask=attention_mask_batch, logits_to_keep=logits_to_keep + 1
            ).logits
            logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
            input_ids_batch = input_ids_batch[:, -logits_to_keep:]
            # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
            # See https://github.com/huggingface/trl/issues/2770
            logits = logits[:, -logits_to_keep:]
            # Divide logits by sampling temperature.
            # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
            logits = logits / self.temperature
            logps = selective_log_softmax(logits, input_ids_batch)  # compute logprobs for the input tokens
            all_logps.append(logps)
        return torch.cat(all_logps, dim=0)

    def _cal_hidden_states(self, model, input_ids, attention_mask):
        # TODO: handle the ddp case: model -> module
        if isinstance(model, StreamModel):
            base_model = model.model.model
        else:
            base_model = model.model
        outputs = base_model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs[0]

    def _compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        advantages = inputs["advantages"]

        hidden_states = self._cal_hidden_states(model, input_ids, attention_mask)
        detached_hidden_states = hidden_states.detach().contiguous().requires_grad_(True)

        loss = torch.tensor(0., device=model.device)
        per_token_kl = []
        coef_1 = []
        per_token_loss = []
        for i in range(0, logits_to_keep, self.chunk_size):
            # calculate the slice of the chunk
            slice_completion = slice(i, min(i + self.chunk_size, logits_to_keep))
            slice_logits = slice(i + prompt_ids.size(1) - 1, min(i + self.chunk_size + prompt_ids.size(1) - 1, input_ids.size(1) - 1))
            slice_input = slice(i + prompt_ids.size(1), min(i + self.chunk_size + prompt_ids.size(1), input_ids.size(1)))

            logits_chunk = self._get_chunk_logits(model, detached_hidden_states, slice_logits)
            logits_chunk = logits_chunk / self.temperature # TODO: check if fp32 logits is required
            input_ids_chunk = input_ids[:, slice_input]
            per_token_logps_chunk = selective_log_softmax(logits_chunk, input_ids_chunk)

            # Compute the KL divergence between the model and the reference model
            if self.beta != 0.0:
                ref_per_token_logps_chunk = inputs["ref_per_token_logps"][:, slice_completion]
                per_token_kl_chunk = (
                    torch.exp(ref_per_token_logps_chunk - per_token_logps_chunk) - (ref_per_token_logps_chunk - per_token_logps_chunk) - 1
                )

            # Compute the loss of the chunk
            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's computation (see
            # _generate_and_score_completions) and use per_token_logps.detach() instead.
            old_per_token_logps_chunk = inputs["old_per_token_logps"][:, slice_completion] if self.num_iterations > 1 else per_token_logps_chunk.detach()
            coef_1_chunk = torch.exp(per_token_logps_chunk - old_per_token_logps_chunk)
            coef_2_chunk = torch.clamp(coef_1_chunk, 1 - self.epsilon_low, 1 + self.epsilon_high)
            per_token_loss1_chunk = coef_1_chunk * advantages.unsqueeze(1)
            per_token_loss2_chunk = coef_2_chunk * advantages.unsqueeze(1)
            per_token_loss_chunk = -torch.min(per_token_loss1_chunk, per_token_loss2_chunk)
            if self.beta != 0.0:
                per_token_loss_chunk = per_token_loss_chunk + self.beta * per_token_kl_chunk

            completion_mask_chunk = completion_mask[:, slice_completion]

            if self.loss_type == "grpo":
                loss_chunk = ((per_token_loss_chunk * completion_mask_chunk).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
            # TODO: check if the implementation of bnpo and dr_grpo is correct
            elif self.loss_type == "bnpo":
                loss_chunk = (per_token_loss_chunk * completion_mask_chunk).sum() / completion_mask.sum().clamp(min=1.0)
            elif self.loss_type == "dr_grpo":
                loss_chunk = (per_token_loss_chunk * completion_mask_chunk).sum() / (per_token_loss_chunk.size(0) * self.max_completion_length)
            else:
                raise ValueError(f"Unknown loss type: {self.loss_type}")

            loss_chunk.backward() # calculate the gradient of hidden_states and lm_head
            
            # record metrics
            loss += loss_chunk.detach()
            per_token_kl.append(per_token_kl_chunk.detach())
            coef_1.append(coef_1_chunk.detach())
            per_token_loss.append(per_token_loss_chunk.detach())
        
        torch.autograd.backward(hidden_states, grad_tensors=detached_hidden_states.grad.detach())
        detached_hidden_states.grad = None
        per_token_kl = torch.cat(per_token_kl, dim=-1)
        coef_1 = torch.cat(coef_1, dim=-1)
        per_token_loss = torch.cat(per_token_loss, dim=-1)
        
        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        if self.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).nanmean().item())

        # Compute the clipped probability ratios
        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = (is_low_clipped * completion_mask).sum() / completion_mask.sum()
        high_clip = (is_high_clipped * completion_mask).sum() / completion_mask.sum()
        clip_ratio = (is_region_clipped * completion_mask).sum() / completion_mask.sum()

        gathered_low_clip = self.accelerator.gather_for_metrics(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
        gathered_high_clip = self.accelerator.gather_for_metrics(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
        gathered_clip_ratio = self.accelerator.gather_for_metrics(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())
        return loss