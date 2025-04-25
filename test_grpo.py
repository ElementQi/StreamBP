# train_grpo.py
import torch
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from fused_grpo_trainer import CustomGRPOTrainer
from transformers import AutoModelForCausalLM
from transformers.trainer_callback import TrainerCallback
from fused_backward_model import StreamModel
import argparse

class GradientMonitorCallback(TrainerCallback):
    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        model = kwargs["model"]
        if isinstance(model, StreamModel):
            model = model.model

        step = state.global_step
        print("========== step", step, "==========")
        print("lm_head.weight.grad", model.lm_head.weight.grad[:5, :5])
        print("q_proj grad", model.model.layers[0].self_attn.q_proj.weight.grad[:5, :5])

        if step == 1:
            print("allocated: ", torch.cuda.memory_allocated() / 2**30, "max allocated: ", torch.cuda.max_memory_allocated() / 2**30)
            quit()

torch.set_printoptions(precision=8)

parser = argparse.ArgumentParser()
parser.add_argument("--chunk_size", type=int, default=500)
parser.add_argument("--mode", type=str, default="stream")
args = parser.parse_args()

dataset = load_dataset("trl-lib/tldr", split="train")

# Define the reward function, which rewards completions that are close to 20 characters
def reward_len(completions, **kwargs):
    return [-abs(20 - len(completion)) for completion in completions]

training_args = GRPOConfig(output_dir="Qwen2.5-0.5B-GRPO", logging_steps=10, num_iterations=15, loss_type="grpo")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")

if args.mode == "stream":
    model = StreamModel(model, gradient_accumulation_steps=1, logits_chunk_size=100, stream_checkpoint=True, checkpoint_chunk_size=args.chunk_size)
    TrainerClass = CustomGRPOTrainer
elif args.mode == "base":
    TrainerClass = GRPOTrainer
else:
    raise ValueError(f"Invalid mode: {args.mode}")

model.gradient_checkpointing_enable()

trainer = TrainerClass(
    model=model,
    reward_funcs=reward_len,
    args=training_args,
    train_dataset=dataset,
    callbacks=[GradientMonitorCallback()],
)
trainer.train()