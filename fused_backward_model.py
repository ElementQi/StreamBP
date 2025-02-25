from transformers import Trainer
from transformers.modeling_utils import PreTrainedModel
from typing import List, Optional, Tuple, Union
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs, _flash_attention_forward
from transformers.processing_utils import Unpack
from transformers.utils import LossKwargs
import torch

class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...

class FusedBackwardModel(torch.nn.Module):
    def __init__(self, model: PreTrainedModel, chunk_size: int):
        # super().__init__(config)
        super().__init__()
        self.chunk_size = chunk_size
        self.model = model

    def __getattr__(self, name):
        """inherit attributes from model"""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        model = self.model # the causal model
 
        outputs = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        B, T, C = hidden_states.size()

        loss = torch.tensor(0., device=hidden_states.device)
        num_chunks = T // self.chunk_size

        for i in range(num_chunks):
            start = i * self.chunk_size
            end = min((i+1)*self.chunk_size, T)
            print(start, end)

            logits_chunk = model.lm_head(hidden_states[:, start:end, :])
            labels_chunk = labels[:, start:end]
            valid_position_num = (labels_chunk != -100).sum().item()

            if valid_position_num == 0:
                continue

            loss_chunk = model.loss_function(logits=logits_chunk, labels=labels_chunk, vocab_size=model.config.vocab_size) * valid_position_num
            loss_chunk.backward(inputs=[hidden_states], retain_graph=True)
            loss += loss_chunk.detach()

        # normalize loss and gradient
        loss /= (labels != -100).sum().item()
        hidden_states.grad = hidden_states.grad / (labels != -100).sum().item()

        torch.autograd.backward(hidden_states, grad_tensors=hidden_states.grad)
        hidden_states.grad = None

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
