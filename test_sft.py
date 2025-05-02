from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from datasets import Dataset
from fused_backward_model import StreamModel
import torch
import argparse
from transformers.trainer_callback import TrainerCallback
from fused_sft_trainer import FusedSFTTrainer

MAX_PAD_RATIO = 0.2
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
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # Create random input sequence
    input_ids = torch.randint(0, tokenizer.vocab_size, (args.num_samples, args.seq_len))

    attention_mask = torch.ones_like(input_ids)
    for mask in attention_mask:
        mask[-torch.randint(1, int(args.seq_len * MAX_PAD_RATIO), (1,)) :] = 0
    
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    
    dataset_dict = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
    
    dataset = Dataset.from_dict(dataset_dict)
    return dataset

parser = argparse.ArgumentParser()
parser.add_argument("--chunk_size", type=int, default=500)
parser.add_argument("--mode", type=str, default="stream")
parser.add_argument("--seq_len", type=int, default=3000)
parser.add_argument("--num_samples", type=int, default=50, help="Number of samples to generate")
parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B", help="Model to use for training")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size")

args = parser.parse_args()

base_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
base_model.train()
dataset = create_dummy_dataset(args)

if args.mode == "stream":
    print("using stream model")
    model = StreamModel(base_model, gradient_accumulation_steps=1, logits_chunk_size=100, checkpoint_chunk_size=args.chunk_size, stream_checkpoint=True)
    model.gradient_checkpointing_enable()
    TrainerClass = FusedSFTTrainer
elif args.mode == "minis":
    print("using minis model")
    from minis.mini_sequence import minisequence
    model = minisequence(base_model, logits_chunk_size=100, chunk_size=args.chunk_size)
    TrainerClass = Trainer
elif args.mode == "base":
    print("using base model with gradient checkpointing")
    base_model.gradient_checkpointing_enable()
    model = base_model
    TrainerClass = Trainer
else:
    raise ValueError(f"Invalid mode: {args.mode}")

training_args = SFTConfig(output_dir="sft",
                          logging_steps=10,
                          per_device_train_batch_size=args.batch_size,
                          gradient_accumulation_steps=1,
                          max_length=None,
                          learning_rate=0., # for gradient profiling
                          )

trainer = TrainerClass(
    model=model,
    train_dataset=dataset,
    args=training_args,
    callbacks=[GradientMonitorCallback()]
)

trainer.train()