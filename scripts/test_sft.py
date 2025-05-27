from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from datasets import Dataset
from streambp.stream_model import StreamModel
import torch
import argparse
import time
import csv
from transformers.trainer_callback import TrainerCallback
from peft import LoraConfig, get_peft_model, PeftModel
from streambp.trainers.stream_sft_trainer import FusedSFTTrainer

MAX_PAD_RATIO = 0.2
torch.set_printoptions(precision=8)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

log_msg = ""

class GradientMonitorCallback(TrainerCallback):
    init_time = None

    def on_epoch_begin(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        self.init_time = time.perf_counter()

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        model = kwargs["model"]
        if isinstance(model, StreamModel):
            model = model.model

        step = state.global_step
        print("========== step", step, "==========")
        print(model.lm_head.weight.grad[:5, :5])
        print(model.model.layers[0].self_attn.q_proj.weight.grad[:5, :5])

        if step == 1:
            torch.cuda.synchronize()
            print("allocated: ", torch.cuda.memory_allocated() / 2**30, "max allocated: ", torch.cuda.max_memory_allocated() / 2**30)
            print("time taken: ", time.perf_counter() - self.init_time)

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
parser.add_argument("--chunk_size", type=int, default=3000)
parser.add_argument("--mode", type=str, default="stream")
parser.add_argument("--seq_len", type=int, default=9000)
parser.add_argument("--num_samples", type=int, default=50, help="Number of samples to generate")
parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B", help="Model to use for training")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
parser.add_argument("--use_lora", action="store_true", help="Use LoRA for training")

lora_config = LoraConfig(
    r=32,
    lora_alpha=128,
    target_modules="all-linear",
    lora_dropout=0.0,
    bias="none",
)

args = parser.parse_args()
log_msg = f"{args.mode}, {args.model_name}, {args.seq_len}, {args.chunk_size}, "

base_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
# base_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32)

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
elif args.mode == "base_no_ckpt":
    print("using base model without gradient checkpointing")
    model = base_model
    TrainerClass = Trainer
else:
    raise ValueError(f"Invalid mode: {args.mode}")

if args.use_lora:
    model = get_peft_model(model, lora_config)

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