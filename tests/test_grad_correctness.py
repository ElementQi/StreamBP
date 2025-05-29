import torch
import pytest
import gc  # Add garbage collection import
from dataclasses import dataclass
from typing import Dict, Any
from transformers import AutoModelForCausalLM
from transformers import Gemma3ForCausalLM


@dataclass
class GradTestConfig:
    vocab_size: int = 128256
    max_pad_ratio: float = 0.2
    # model_name: str = "Qwen/Qwen3-4B"
    # model_name: str = "Qwen/Qwen3-8B"
    # model_name: str = "Qwen/Qwen2.5-7B"
    # model_name: str = "Qwen/Qwen2.5-3B"
    # model_name: str = "meta-llama/Meta-Llama-3-8B"
    # model_name: str = "meta-llama/Llama-3.1-8B"
    # model_name: str = "meta-llama/Llama-3.2-1B"
    # model_name: str = "meta-llama/Llama-3.2-3B"
    model_name: str = "google/gemma-3-1b-pt"
    device: str = "cuda"
    seq_len: int = 5000
    batch_size: int = 1
    abs_tolerance: float = 1e-6  # Absolute error tolerance
    rel_tolerance: float = 0.1  # 10% relative error tolerance (generous for bf16)
    epsilon: float = 1e-10  # Small constant to avoid division by zero
    fp32_max_rel_error: float = 0.005  # 0.5% relative error for FP32 models


class ModelSetup:
    def __init__(self, config: GradTestConfig):
        self.config = config
        self._set_seeds()

    def _set_seeds(self) -> None:
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)

    def create_test_data(self) -> Dict[str, torch.Tensor]:
        input_ids = torch.randint(
            0, self.config.vocab_size, (self.config.batch_size, self.config.seq_len)
        ).to(self.config.device)

        attention_mask = torch.ones_like(input_ids)
        pad_length = int(self.config.seq_len * self.config.max_pad_ratio)
        attention_mask[:, -pad_length:] = 0

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def create_base_model(self, dtype: torch.dtype) -> AutoModelForCausalLM:
        return (
            AutoModelForCausalLM.from_pretrained(
                self.config.model_name, torch_dtype=dtype
            )
            .to(self.config.device)
            .train()
        )

    def create_stream_model(self, base_model: AutoModelForCausalLM):
        # Conditionally import StreamModel based on model type
        if isinstance(base_model, Gemma3ForCausalLM):
            from streambp.stream_model_gemma import StreamModel
        else:
            # For Qwen3ForCausalLM, Qwen2ForCausalLM, LlamaForCausalLM and others
            from streambp.stream_model import StreamModel

        return StreamModel(
            base_model,
            gradient_accumulation_steps=1,
            logits_chunk_size=100,
            checkpoint_chunk_size=500,
            stream_checkpoint=True,
        )


class GradientExtractor:
    GRAD_KEYS = ["q_proj", "k_proj", "lm_head"]

    @staticmethod
    def extract_gradients(
        model: Any, inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        model.zero_grad()
        output = model(**inputs, use_cache=False, return_dict=True)

        if output.loss.requires_grad:
            output.loss.backward()

        # Handle StreamModel wrapper vs base model
        if hasattr(model.model, "model"):
            # StreamModel case
            causal_model = model.model
        else:
            # Base model case
            causal_model = model

        # Access the first layer directly from the model
        layer = causal_model.model.layers[0]

        return {
            "q_proj": layer.self_attn.q_proj.weight.grad.detach().clone(),
            "k_proj": layer.self_attn.k_proj.weight.grad.detach().clone(),
            "lm_head": causal_model.lm_head.weight.grad.detach().clone(),
        }

    @staticmethod
    def calculate_absolute_error(
        reference: torch.Tensor, comparison: torch.Tensor
    ) -> float:
        """Calculate mean absolute error as defined in the paper."""
        return torch.mean(torch.abs(reference - comparison.to(reference.dtype))).item()

    @staticmethod
    def calculate_relative_error(
        reference: torch.Tensor, comparison: torch.Tensor, epsilon: float = 1e-10
    ) -> float:
        """Calculate mean relative error as defined in the paper."""
        abs_diff = torch.abs(reference - comparison.to(reference.dtype))
        abs_ref = torch.abs(reference) + epsilon
        return torch.mean(abs_diff / abs_ref).item()

    @staticmethod
    def calculate_error(reference: torch.Tensor, comparison: torch.Tensor) -> float:
        return torch.mean(torch.abs(reference - comparison.to(reference.dtype))).item()


@pytest.fixture(scope="module")
def test_setup():
    config = GradTestConfig()
    setup = ModelSetup(config)
    test_data = setup.create_test_data()

    return {"config": config, "test_data": test_data, "setup": setup}


def test_stream_vs_base_gradient_precision(test_setup):
    """Test that StreamModel gradients are within acceptable error bounds compared to base models."""
    config = test_setup["config"]
    test_data = test_setup["test_data"]
    setup = test_setup["setup"]

    extractor = GradientExtractor()

    # Process models sequentially to reduce memory usage
    model_configs = [
        ("base_fp32", torch.float32, False),
        ("base_bf16", torch.bfloat16, False),
        ("stream_bf16", torch.bfloat16, True),
        ("stream_fp32", torch.float32, True),
    ]

    # Store reference gradients and base bf16 errors separately
    reference_grads = None
    base_bf16_errors = None

    for model_name, dtype, is_stream in model_configs:
        # Reset memory stats before each model
        torch.cuda.reset_peak_memory_stats()

        # Create model
        base_model = setup.create_base_model(dtype)
        model = setup.create_stream_model(base_model) if is_stream else base_model
        model.gradient_checkpointing_enable()

        # Extract gradients
        current_grads = extractor.extract_gradients(model, test_data)

        # Synchronize CUDA operations and print peak memory
        torch.cuda.synchronize()
        peak_memory_gb = torch.cuda.max_memory_allocated() / 2**30
        print(f"{model_name} max allocated: {peak_memory_gb:.3f} GB")

        # Store reference (first model) or compare immediately
        if model_name == "base_fp32":
            reference_grads = current_grads
        elif model_name == "base_bf16":
            # Store base bf16 errors for later comparison with stream bf16
            base_bf16_errors = _compare_gradients(
                reference_grads, current_grads, model_name, extractor, config
            )
            del current_grads
        else:
            # Perform comparison immediately to avoid memory accumulation
            _compare_gradients(
                reference_grads,
                current_grads,
                model_name,
                extractor,
                config,
                base_bf16_errors,
            )

            # Clear current gradients immediately after comparison
            del current_grads

        # Clean up model immediately
        del model, base_model
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # Ensure CUDA operations complete
        gc.collect()  # Force garbage collection


def _compare_gradients(
    reference_grads, current_grads, model_name, extractor, config, base_bf16_errors=None
):
    """Helper function to compare gradients and perform assertions."""
    current_errors = {}

    for key in extractor.GRAD_KEYS:
        if model_name == "base_bf16":
            base_abs_error = extractor.calculate_absolute_error(
                reference_grads[key], current_grads[key]
            )
            base_rel_error = extractor.calculate_relative_error(
                reference_grads[key], current_grads[key], config.epsilon
            )
            print(f"Gradient errors for {key}:")
            print(
                f"  Base BF16    - Abs: {base_abs_error:.8e}, Rel: {base_rel_error:.4%}"
            )

            current_errors[key] = {"abs": base_abs_error, "rel": base_rel_error}

        elif model_name == "stream_bf16":
            stream_bf16_abs_error = extractor.calculate_absolute_error(
                reference_grads[key], current_grads[key]
            )
            stream_bf16_rel_error = extractor.calculate_relative_error(
                reference_grads[key], current_grads[key], config.epsilon
            )
            print(
                f"  Stream BF16  - Abs: {stream_bf16_abs_error:.8e}, Rel: {stream_bf16_rel_error:.4%}"
            )

            # Stream BF16 should have similar precision to base bf16 model
            base_rel_error = base_bf16_errors[key]["rel"]
            base_abs_error = base_bf16_errors[key]["abs"]
            max_allowed_rel_error = max(base_rel_error * 1.5, config.rel_tolerance)
            max_allowed_abs_error = max(base_abs_error * 1.5, config.abs_tolerance)

            assert stream_bf16_rel_error <= max_allowed_rel_error, (
                f"Stream BF16 relative error ({stream_bf16_rel_error:.4%}) exceeds "
                f"maximum allowed ({max_allowed_rel_error:.4%}) for {key}"
            )

            assert stream_bf16_abs_error <= max_allowed_abs_error, (
                f"Stream BF16 absolute error ({stream_bf16_abs_error:.8e}) exceeds "
                f"maximum allowed ({max_allowed_abs_error:.8e}) for {key}"
            )

            print(f"  ✓ {key} gradients within acceptable bounds")

        elif model_name == "stream_fp32":
            stream_fp32_abs_error = extractor.calculate_absolute_error(
                reference_grads[key], current_grads[key]
            )
            stream_fp32_rel_error = extractor.calculate_relative_error(
                reference_grads[key], current_grads[key], config.epsilon
            )
            print(
                f"  Stream FP32  - Abs: {stream_fp32_abs_error:.8e}, Rel: {stream_fp32_rel_error:.4%}"
            )

            # Stream FP32 should have very high precision
            assert stream_fp32_rel_error <= config.fp32_max_rel_error, (
                f"Stream FP32 relative error ({stream_fp32_rel_error:.4%}) exceeds "
                f"{config.fp32_max_rel_error * 100}% tolerance for {key} - should match paper's high precision results"
            )
            print(f"  ✓ {key} gradients within acceptable bounds")

    return current_errors if model_name == "base_bf16" else None
