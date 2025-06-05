from transformers import Trainer
from transformers.modeling_utils import PreTrainedModel
from typing import List, Optional, Tuple, Union, Any, Dict
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs, _flash_attention_forward
from transformers.processing_utils import Unpack
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, LlamaRMSNorm, repeat_kv, rotate_half
from transformers.utils import LossKwargs
import inspect
import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import check_backward_validity, _infer_device_type, _get_autocast_kwargs, _get_device_module, get_device_states, detach_variable

class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...

global_dict = {}
stream_buffer = {}

def apply_rotary_pos_emb(states, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    states_embed = (states * cos) + (rotate_half(states) * sin)
    return states_embed

class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        # self.act_fn = ACT2FN[config.hidden_act]
        self.act_fn = nn.SiLU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

class CheckpointFunctionForStreamBackward(torch.autograd.Function):
    chunk_size: int = 100
    @staticmethod
    def forward(ctx, run_function, preserve_rng_state, *args):
        check_backward_validity(args)
        ctx.run_function = run_function
        ctx.preserve_rng_state = preserve_rng_state
        # Accommodates the (remote) possibility that autocast is enabled for cpu AND gpu.
        ctx.device = _infer_device_type(*args)
        ctx.device_autocast_kwargs, ctx.cpu_autocast_kwargs = _get_autocast_kwargs(
            ctx.device
        )
        ctx.chunk_size = CheckpointFunctionForStreamBackward.chunk_size
        if preserve_rng_state:
            ctx.fwd_cpu_state = torch.get_rng_state()
            # Don't eagerly initialize the cuda context by accident.
            # (If the user intends that the context is initialized later, within their
            # run_function, we SHOULD actually stash the cuda state here.  Unfortunately,
            # we have no way to anticipate this will happen before we run the function.)
            ctx.had_device_in_fwd = False
            device_module = _get_device_module(ctx.device)
            if getattr(device_module, "_initialized", False):
                ctx.had_device_in_fwd = True
                ctx.fwd_devices, ctx.fwd_device_states = get_device_states(*args)

        # Save non-tensor inputs in ctx, keep a placeholder None for tensors
        # to be filled out during the backward.
        ctx.inputs = []
        ctx.tensor_indices = []
        tensor_inputs = []
        for i, arg in enumerate(args):
            if torch.is_tensor(arg):
                tensor_inputs.append(arg)
                ctx.tensor_indices.append(i)
                ctx.inputs.append(None)
            else:
                ctx.inputs.append(arg)

        ctx.save_for_backward(*tensor_inputs)

        with torch.no_grad():
            outputs = run_function(*args)
        return outputs

    @staticmethod
    def backward(ctx, *args):
        # Copy the list to avoid modifying original list.
        inputs = list(ctx.inputs)
        tensor_indices = ctx.tensor_indices
        tensors = ctx.saved_tensors
        device_module = _get_device_module(ctx.device)

        # Fill in inputs with appropriate saved tensors.
        for i, idx in enumerate(tensor_indices):
            inputs[idx] = tensors[i]

        detached_inputs = detach_variable(tuple(inputs))

        hidden_states_grad = args[0] # unpack args
        num_chunks = math.ceil(hidden_states_grad.size(1) / ctx.chunk_size)

        if "zero2_optimizer" in global_dict:
            global_dict["zero2_optimizer"].process_gradients = lambda *args, **kwargs: None

        for i in range(num_chunks):
            start = i * ctx.chunk_size
            end = min((i+1)*ctx.chunk_size, hidden_states_grad.size(1))

            if (i == num_chunks - 1) and "zero2_optimizer" in global_dict:
                global_dict["zero2_optimizer"].process_gradients = global_dict["zero2_gradient_process_func"]
            with torch.enable_grad():
                outputs = ctx.run_function(*detached_inputs, chunk_range=(start, end)) # TODO: make it more elegant
                if isinstance(outputs, tuple):
                    hidden_states = outputs[0]
                else:
                    hidden_states = outputs
                torch.autograd.backward(
                        hidden_states,
                        grad_tensors=hidden_states_grad[:, start:end, :].detach(), 
                        retain_graph=True if end < hidden_states_grad.size(1) else False
                    )


        grads = tuple(
            inp.grad if isinstance(inp, torch.Tensor) else None
            for inp in detached_inputs
        )

        return (None, None) + grads

class StreamMLP(nn.Module):
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp

    def __getattr__(self, name):
        """inherit attributes"""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.mlp, name)

    def forward(self, x, chunk_range=None):
        if chunk_range is not None:
            x = x[:, chunk_range[0]:chunk_range[1], :]
        down_proj = self.mlp(x)
        return down_proj

class StreamDecoderLayer(nn.Module):
    def __init__(self, base_layer):
        super().__init__()
        self.base_layer = base_layer
        self._setup_attn()

    def _setup_attn(self):
        """enable stream forward"""
        self.base_layer.self_attn = StreamAttention(self.base_layer.self_attn)

    def __getattr__(self, name):
        """inherit attributes"""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_layer, name)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        chunk_range: Optional[Tuple[int, int]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            chunk_range (`Tuple[int, int]`, *optional*): chunk range for calculating query states
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """

        residual = hidden_states
        
        if chunk_range is not None:
            residual = hidden_states[:, chunk_range[0]:chunk_range[1], :]

        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            chunk_range=chunk_range,
            **kwargs,
        )

        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

class StreamAttention(torch.nn.Module):

    def __init__(self, self_attn):
        super().__init__()
        self.self_attn = self_attn
        self.cache_states = {}
        self._setup_stream_buffer()

    def __getattr__(self, name):
        """inherit attributes"""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.self_attn, name)

    def _setup_stream_buffer(self):
        for model in stream_buffer:
            if any(m is self.self_attn for m in model.modules()):
                self.stream_buffer = stream_buffer[model]
                return

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        chunk_range: Optional[Tuple[int, int]] = None,
        key_states: Optional[torch.Tensor] = None,
        value_states: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()
        if chunk_range is not None:
            chunk_startidx, chunk_endidx = chunk_range
            chunk_len = chunk_endidx - chunk_startidx
        else:
            chunk_startidx, chunk_endidx, chunk_len = 0, q_len, q_len

        key_states = self.k_proj(hidden_states[:, :chunk_endidx, :])
        value_states = self.v_proj(hidden_states[:, :chunk_endidx, :])

        # Only compute query states for the current chunk
        query_states = self.q_proj(hidden_states[:, chunk_startidx:chunk_endidx, :])

        # Compute RoPE embeddings efficiently
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings

        query_states = query_states.view(bsz, chunk_len, -1, self.head_dim).transpose(1, 2)

        key_states = key_states.view(bsz, chunk_endidx, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, chunk_endidx, -1, self.head_dim).transpose(1, 2)

        if hasattr(self, "q_norm"):
            query_states = self.q_norm(query_states)
        if hasattr(self, "k_norm"):
            key_states = self.k_norm(key_states)

        # Apply RoPE efficiently
        key_states = apply_rotary_pos_emb(key_states, cos[:, :chunk_endidx], sin[:, :chunk_endidx])

        query_states = apply_rotary_pos_emb(query_states, cos[:, chunk_startidx:chunk_endidx], sin[:, chunk_startidx:chunk_endidx])

        # TODO: check the meaning of this section
        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Optimize attention mask handling
        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]
            # causal_mask = causal_mask[:, :, chunk_startidx:chunk_endidx, :]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        is_causal = True if causal_mask is None and q_len > 1 else False

        # causal_mask = causal_mask[:, :, chunk_startidx:chunk_endidx, :]

        # TODO: check corner case
        if query_states.shape[2] == 1 and causal_mask is None:
            # Generation mode, the new state will attend to all the previous states
            is_causal = False
        elif self.stream_buffer["attention_mask"] is None:
            causal_mask = None
            is_causal = True
        else:
            causal_mask = self._generate_causal_mask(chunk_startidx, chunk_endidx, query_states.dtype, query_states.device)
            is_causal = False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            scale=self.scaling,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous() # TODO: check
        attn_output = attn_output.view(bsz, chunk_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value

    def _generate_causal_mask(self, start_idx, end_idx, dtype, device):
        # TODO: handle sliding window attention
        sub_attention_mask = self.stream_buffer["attention_mask"][:, :end_idx]
        batch_size = sub_attention_mask.shape[0]

        min_dtype = torch.finfo(dtype).min
        chunk_len = end_idx - start_idx
        causal_mask = torch.full((batch_size, 1, chunk_len, end_idx), fill_value=min_dtype, dtype=dtype, device=device)
        active_mask = torch.arange(start_idx, end_idx, device=device).view(-1, 1) >= torch.arange(end_idx, device=device)
        causal_mask.masked_fill_(active_mask, 0.)

        zero_mask_indices = (sub_attention_mask == 0).unsqueeze(1).unsqueeze(1)
        causal_mask.masked_fill_(zero_mask_indices, min_dtype)

        # For numerical stability; see https://github.com/pytorch/pytorch/issues/110213 for full details
        # causal_mask.mul(~torch.all(causal_mask == min_dtype, dim=-1, keepdim=True))

        return causal_mask

class StreamModel(torch.nn.Module):
    def __init__(self, model: PreTrainedModel, gradient_accumulation_steps, gradient_accumulation_mode="sum", logits_chunk_size: int=500, stream_checkpoint: bool=True, checkpoint_chunk_size: int=500):
        """ The StreamModel class wraps the original model to save the memory usage. """
        super().__init__()
        self.logits_chunk_size = logits_chunk_size
        self.stream_checkpoint = stream_checkpoint
        self.checkpoint_chunk_size = checkpoint_chunk_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_accumulation_mode = gradient_accumulation_mode
        self.model = model

        # enable transformer layer to forward in chunk mode, i.e. calculate the query states
        # of a particular chunk (NOTE: the key and value states are still calculated for the whole sequence)
        self._setup_stream_buffer()
        self._setup_stream_forward()
        self._setup_gradient_accumulation()

        # if stream_checkpoint:
        #     # when calculating the gradient for a checkpointed layer, re-forward and backward in stream mode
        #     self.gradient_checkpointing_enable(checkpoint_chunk_size=checkpoint_chunk_size)

    def _setup_stream_buffer(self):
        if self not in stream_buffer:
            stream_buffer[self] = {}
        self.stream_buffer = stream_buffer[self]

        # set up base model's stream buffer
        base_model = self.get_base_model(self.model)
        base_model.stream_buffer = self.stream_buffer
        def _attention_mask_recording_wrapper(func, *args, **kwargs):
            def wrapped_func(*args, **kwargs):
                if "attention_mask" in kwargs and kwargs["attention_mask"] is not None:
                    self.stream_buffer["attention_mask"] = kwargs["attention_mask"]
                return func(*args, **kwargs)
            return wrapped_func
        base_model.forward = _attention_mask_recording_wrapper(base_model.forward)

    def _setup_stream_forward(self):
        # TODO: add check for layer type
        # TODO: avoid modifying the base model's behavior
        for i in range(len(self.model.model.layers)):
            self.model.model.layers[i] = StreamDecoderLayer(self.model.model.layers[i])

    def _setup_gradient_accumulation(self):
        self._cur_gradient_accumulation_step = 0
        self._valid_pos_num = 0

    def get_base_model(self, model):
        is_base_model = False
        while not is_base_model:
            for attr in ["model", "base_model", "module"]:
                if hasattr(model, attr) and not (getattr(model, attr) is model):
                    model = getattr(model, attr)
                else:
                    is_base_model = True
                    break
        return model

    def __getattr__(self, name):
        """inherit attributes from model"""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def gradient_checkpointing_enable(self: "PreTrainedModel", gradient_checkpointing_kwargs: Optional[Dict[str, Any]] = None):
        r"""
        Activates gradient checkpointing for the current model.

        Modification of the original method to enable gradient checkpointing for block-wise optimizer.
        """
        from functools import partial

        if not self.supports_gradient_checkpointing:
            raise ValueError("{} does not support gradient checkpointing.".format(self.__class__.__name__))

        if self.stream_checkpoint:
            # TODO: handle the argument of use_reentrant
            def stream_gradient_checkpointing_func(func, *args, **kwargs):
                CheckpointFunctionForStreamBackward.chunk_size = self.checkpoint_chunk_size
                preserve = kwargs.pop("preserve_rng_state", True)
                return CheckpointFunctionForStreamBackward.apply(func, preserve, *args, **kwargs)
            
            gradient_checkpointing_func = stream_gradient_checkpointing_func
        else:
            if gradient_checkpointing_kwargs is None:
                gradient_checkpointing_kwargs = {"use_reentrant": True}
            from torch.utils.checkpoint import checkpoint
            gradient_checkpointing_func = partial(checkpoint, **gradient_checkpointing_kwargs)

        def custom_gradient_checkpointing_func(func, *args, **kwargs):
            # in case that the func is a partial function
            if hasattr(func, "func"):
                func = func.func
            module: "torch.nn.Module" = func.__self__

            if any(param.requires_grad for param in module.parameters()):
                for arg in args:
                    if torch.is_tensor(arg) and torch.is_floating_point(arg):
                        arg.requires_grad_(True)
                        break # TODO: for avoiding enabling gradient of causal_mask; need to make it more elegant

            return gradient_checkpointing_func(func, *args, **kwargs)

        if "value" in inspect.signature(self._set_gradient_checkpointing).parameters:  # old GC format
            self.apply(partial(self._set_gradient_checkpointing, value=True))
            self.enable_input_require_grads()
        else:  # have already enabled input require gradients
            self._set_gradient_checkpointing(enable=True, gradient_checkpointing_func=custom_gradient_checkpointing_func)

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
        
        if "zero2_optimizer" in global_dict:
            global_dict["zero2_gradient_process_func"] = global_dict["zero2_optimizer"].process_gradients
            global_dict["zero2_optimizer"].process_gradients = lambda *args, **kwargs: None

        if attention_mask is not None:
            self.stream_buffer["attention_mask"] = attention_mask
            # avoid creating a T by T mask. Only generate the attention mask of the partition during self-attention
            attention_mask = None

        if (not self.training) or (not torch.is_grad_enabled()):
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                num_logits_to_keep=num_logits_to_keep,
                **kwargs,
            )        

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
        num_chunks = math.ceil(T / self.logits_chunk_size)

        detached_hidden_states = hidden_states.detach().contiguous().requires_grad_(True)

        if model.lm_head.weight.grad is None:
            model.lm_head.weight.grad = torch.zeros_like(model.lm_head.weight)

        for i in range(num_chunks):
            start = i * self.logits_chunk_size
            end = min((i+1)*self.logits_chunk_size+1, T)

            logits_chunk = model.lm_head(detached_hidden_states[:, start:end, :])
            labels_chunk = labels[:, start:end]

            # TODO: check correctness
            chunk_valid_posnum = (labels_chunk != -100).sum().item() - 1 # -1 for the last token

            if chunk_valid_posnum <= 0:
                continue

            loss_chunk = model.loss_function(logits=logits_chunk, labels=labels_chunk, vocab_size=model.config.vocab_size) * chunk_valid_posnum
            loss_chunk.backward()
            # loss_chunk.backward(inputs=[detached_hidden_states, logits_chunk])

            # # This helps avoids storing two copies of lm_head's gradient
            # # When using ZeRO-2, lm_head's gradient will be reduced in the end of backward pass, i.e. in _backward_epilogue function function
            # with torch.no_grad():
            #     logits_chunk_grad = logits_chunk.grad.view(-1, model.config.vocab_size).T
            #     logits_chunk_grad = logits_chunk_grad.to(model.lm_head.weight.grad.dtype)
            #     model.lm_head.weight.grad.addmm_(
            #         logits_chunk_grad,
            #         detached_hidden_states[:, start:end, :].reshape(-1, C),
            #     )

            # del logits_chunk.grad
            del logits_chunk
            loss += loss_chunk.detach()

        # normalize loss and gradient
        batch_valid_posnum = (labels != -100).sum().item()
        loss.div_(batch_valid_posnum)
        # detached_hidden_states.grad.div_(batch_valid_posnum)
        # model.lm_head.weight.grad.div_(batch_valid_posnum)

        # with torch.amp.autocast(device_type="cuda", enabled=False):
        #     # If enabled, gradient of bf16 operator related weights will not be computed.
        #     # TODO: figure out why and check if there slows down the backward
        #     torch.autograd.backward(hidden_states, grad_tensors=detached_hidden_states.grad.detach())
        torch.autograd.backward(hidden_states, grad_tensors=detached_hidden_states.grad.detach())
        
        detached_hidden_states.grad = None

        self._cur_gradient_accumulation_step += 1
        self._valid_pos_num += batch_valid_posnum
        if self._cur_gradient_accumulation_step == self.gradient_accumulation_steps:
            for param in self.parameters():
                if param.grad is not None:
                    param.grad.div_(self._valid_pos_num)
            self._cur_gradient_accumulation_step = 0
            self._valid_pos_num = 0

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )