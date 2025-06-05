from trl import DPOConfig, DPOTrainer
from ..stream_model import StreamModel
import torch
import torch.nn as nn
from typing import Any, Union
from transformers import PreTrainedModel
from trl.trainer.utils import selective_log_softmax

def get_base_model(model):
    is_base_model = False
    while not is_base_model:
        for attr in ["model", "base_model", "module"]:
            if hasattr(model, attr) and not (getattr(model, attr) is model):
                model = getattr(model, attr)
            else:
                is_base_model = True
                break
    return model

def get_causal_model(model):
    is_causal_model = False
    while not is_causal_model:
        if hasattr(model, "lm_head"):
            is_causal_model = True
            break
        for attr in ["model", "base_model", "module"]:
            if hasattr(model, attr) and not (getattr(model, attr) is model):
                model = getattr(model, attr)
                break
    return model

class StreamDPOTrainer(DPOTrainer):
    def __init__(
            self, *args, **kwargs
        ) -> None:
        # assert isinstance(kwargs["model"], StreamModel), "model must be a StreamModel"
        self.chunk_size = kwargs["model"].logits_chunk_size
        self.vocab_size = kwargs["model"].model.lm_head.out_features
        # self.max_completion_length = kwargs.pop("max_completion_length") # NOTE: for generating dummy data only; to delete after testing
        super().__init__(**kwargs)
        self.accelerator.backward = lambda loss: None # backward is fused with forward, no need to call accelerator.backward
    
    @torch.no_grad()
    def _cal_correction_factor(self, model, ref_model, hidden_states, input_ids, attention_mask, num_examples=None):
        """
        Define log ratio c:= (log p_w/p_l - log pref_w/pref_l)
        The DPO loss is log(sigmoid(beta*c))
        The gradient w.r.t. policy model pi is (1-sigmoid(beta*c))*beta*(partial c/partial pi)
        Define:
            corr_factor := (1-sigmoid(beta*c))*beta
            c_i := log p_w(i)/p_l(i) - log pref_w(i)/pref_l(i), which is chunked log ratio
        The gradient w.r.t. policy model pi is corr_factor * \sum_i (partial c_i/partial pi).

        Input:
            hidden_states (2B, T_completion, d): hidden states of **completion part**. The first half is chosen, the second half is rejected.
            input_ids (2B, T_completion)
            attention_mask (2B, T_completion)
            num_examples: number of examples in the batch.
        """

        if num_examples is None:
            num_examples = hidden_states.shape[0] // 2
            assert hidden_states.shape[0] % 2 == 0

        completion_len = hidden_states.shape[1]
        log_ratios = []
        for i in range(0, completion_len, self.chunk_size):
            logits_slice = slice(i, min(i + self.chunk_size, completion_len-1))
            label_slice = slice(i + 1, min(i + self.chunk_size+1, completion_len))
            chunked_logdiff = self._cal_logdiff(get_causal_model(model), hidden_states[:, logits_slice, :], input_ids[:, label_slice], attention_mask[:, label_slice], num_examples)
            if ref_model is not None:
                chunked_ref_logdiff = self._cal_logdiff(get_causal_model(ref_model), hidden_states[:, logits_slice, :], input_ids[:, label_slice], attention_mask[:, label_slice], num_examples)
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    # TODO: check correctness
                    chunked_ref_logdiff = self._cal_logdiff(get_causal_model(self.model), hidden_states[:, logits_slice, :], input_ids[:, label_slice], attention_mask[:, label_slice], num_examples)
            log_ratios.append(chunked_logdiff.sum(dim=-1) - chunked_ref_logdiff.sum(dim=-1)) # c_i
        
        log_ratios = torch.vstack(log_ratios).sum(dim=0)
        corr_factor = (torch.sigmoid(self.beta * log_ratios) - 1) * self.beta
        losses = - torch.nn.functional.logsigmoid(log_ratios)
        return corr_factor, losses # (B, )

    def _cal_logdiff(self, model, hidden_states, label, attention_mask, num_examples):
        logits = model.lm_head(hidden_states)
        logps = selective_log_softmax(logits, label)
        logps = logps * attention_mask
        chosen_logps = logps[:num_examples]
        rejected_logps = logps[num_examples:]
        return chosen_logps - rejected_logps # (B, T)

    def _fused_dpo_backward_single_example(self, model, inputs):
        """
        For single example case (i.e. batch size = 1), there is no need to calculate 
        correction factor in advance.
        """
        pass # TODO

    def _fused_dpo_backward(self, model, inputs):
        num_examples = inputs["prompt_input_ids"].shape[0]
        
        # if num_examples == 1:
        #     return self._fused_dpo_backward_single_example(model, inputs)
        
        inputs = self.concatenated_inputs(inputs, padding_value=self.padding_value)
        prompt_input_ids = inputs["prompt_input_ids"]
        prompt_attention_mask = inputs["prompt_attention_mask"]
        completion_input_ids = inputs["completion_input_ids"]
        completion_attention_mask = inputs["completion_attention_mask"]
        prompt_len = prompt_input_ids.shape[1]

        input_ids = torch.cat((prompt_input_ids, completion_input_ids), dim=1)
        attention_mask = torch.cat((prompt_attention_mask, completion_attention_mask), dim=1)

        hidden_states = self._cal_hidden_states(model, input_ids, attention_mask)
        detached_hidden_states = hidden_states.detach().contiguous().requires_grad_(True)

        corr_factor, losses = self._cal_correction_factor(model=model,
                                                  ref_model=self.ref_model,
                                                  hidden_states=detached_hidden_states[:, prompt_len:, :],
                                                  input_ids=completion_input_ids,
                                                  attention_mask=completion_attention_mask,
                                                  num_examples=num_examples
                                                  )
        
        for i in range(prompt_len, input_ids.shape[1], self.chunk_size):
            input_slice = slice(i, min(i + self.chunk_size, input_ids.shape[1]-1))
            label_slice = slice(i + 1, min(i + self.chunk_size + 1, input_ids.shape[1]))
            
            chunked_logdiff = self._cal_logdiff(model,
                                                 detached_hidden_states[:, input_slice, :],
                                                 input_ids[:, label_slice],
                                                 attention_mask[:, label_slice],
                                                 num_examples
                                                 )
            corrected_chunk_logdiff = corr_factor * chunked_logdiff.sum(dim=-1)
            corrected_chunk_logdiff.mean().backward()

        torch.autograd.backward(hidden_states, grad_tensors=detached_hidden_states.grad.detach())
        detached_hidden_states.grad = None

        return losses.mean(), {}

    def _cal_logps(self, model, hidden_states, labels, num_examples=None):
        if num_examples is None:
            num_examples = hidden_states.shape[0] // 2
            assert hidden_states.shape[0] % 2 == 0
        logits = model.lm_head(hidden_states)
        per_token_logps = selective_log_softmax(logits, labels)
        chosen_per_token_logps = per_token_logps[:num_examples]
        rejected_per_token_logps = per_token_logps[num_examples:]
        chosen_logps = chosen_per_token_logps.sum(-1)
        rejected_logps = rejected_per_token_logps.sum(-1)
        return chosen_per_token_logps, rejected_per_token_logps, chosen_logps, rejected_logps

    def _cal_hidden_states(self, model, input_ids, attention_mask):
        model = get_base_model(model)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs[0]
    
    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
        num_items_in_batch=None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, dict[str, torch.Tensor]]]:
        # device_type = "xpu" if is_torch_xpu_available() else "cuda"
        # compute_loss_context_manager = (
        #     amp.autocast(device_type) if self._peft_has_been_casted_to_bf16 else nullcontext()
        # )
        # with compute_loss_context_manager:
        
        # loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")
        loss, metrics = self._fused_dpo_backward(model, inputs)

        # Make sure to move the loss to the device the original accumulating loss is at back in the `Trainer` class:
        loss = loss.to(self.args.device)
        # force log the metrics
        self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return loss, metrics
        print("loss", loss)

        return loss