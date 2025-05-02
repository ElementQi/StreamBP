# train_grpo.py
import torch
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from fused_grpo_trainer import CustomGRPOTrainer, OriginalGRPOTrainer
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_callback import TrainerCallback
from fused_backward_model import StreamModel
import argparse

torch.set_printoptions(precision=8)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

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

def create_dummy_dataset(args):
    # Configuration
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Create random input sequence
    input_ids = torch.randint(0, tokenizer.vocab_size, (args.num_samples, args.prompt_len))
    
    # NOTE: no attention mask for the prompt
    attention_mask = torch.ones_like(input_ids)

    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    # NOTE: For memory profiling, we directly use the completion_ids rather than generating them for faster experiment
    # See CustomGRPOTrainer._prepare_inputs for more details
    dataset_dict = {
        "prompt_ids": input_ids,
        "prompt_mask": attention_mask,
        "labels": labels,
    }
    
    dataset = Dataset.from_dict(dataset_dict)
    return dataset

parser = argparse.ArgumentParser()
parser.add_argument("--chunk_size", type=int, default=500)
parser.add_argument("--mode", type=str, default="stream")
parser.add_argument("--max_completion_len", type=int, default=1000, help="Sequence length for the completion")
parser.add_argument("--prompt_len", type=int, default=1000, help="Sequence length for the prompt")
parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples to generate")
parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B", help="Model to use for training")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
args = parser.parse_args()

# dataset = load_dataset("trl-lib/tldr", split="train")
dataset = create_dummy_dataset(args)

# Define the reward function, which rewards completions that are close to 20 characters
def reward_len(completions, **kwargs):
    return [-abs(20 - len(completion)) for completion in completions]

training_args = GRPOConfig(output_dir="Qwen2.5-0.5B-GRPO",
                           logging_steps=10, 
                           num_iterations=15,
                           loss_type="grpo",
                           per_device_train_batch_size=args.batch_size,
                           gradient_accumulation_steps=1,
                           max_completion_length=args.max_completion_len,
                           )
model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)

if args.mode == "stream":
    model = StreamModel(model, gradient_accumulation_steps=1, logits_chunk_size=100, stream_checkpoint=True, checkpoint_chunk_size=args.chunk_size)
    TrainerClass = CustomGRPOTrainer
elif args.mode == "base":
    TrainerClass = OriginalGRPOTrainer # The original GRPO trainer that enables dummy data
else:
    raise ValueError(f"Invalid mode: {args.mode}")

kwargs = {
    "max_completion_length": args.max_completion_len,
}
model.gradient_checkpointing_enable()

trainer = TrainerClass(
    model=model,
    reward_funcs=reward_len,
    args=training_args,
    train_dataset=dataset,
    callbacks=[GradientMonitorCallback()],
    **kwargs
)
trainer.train()