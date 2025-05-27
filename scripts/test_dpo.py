from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_callback import TrainerCallback
from streambp.stream_model import StreamModel
from streambp.trainers.stream_dpo_trainer import FusedDPOTrainer
from peft import LoraConfig, get_peft_model, PeftModel
import argparse
from datasets import Dataset
import torch

torch.set_printoptions(precision=8)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

MAX_PAD_RATIO = 0.2

class GradientMonitorCallback(TrainerCallback):
    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        model = kwargs["model"]
        if isinstance(model, StreamModel):
            model = model.model

        step = state.global_step
        print("========== step", step, "==========")
        print(model.lm_head.weight.grad[:5, :5])
        print(model.model.layers[0].self_attn.q_proj.weight.grad[:5, :5])

        if step == 1:
            print("allocated: ", torch.cuda.memory_allocated() / 2**30, "max allocated: ", torch.cuda.max_memory_allocated() / 2**30)
            quit()

def create_dummy_dataset(args):
    # Configuration
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Create random input sequence
    prompt_ids = torch.randint(0, tokenizer.vocab_size, (args.num_samples, args.prompt_len))
    chosen_ids = torch.randint(0, tokenizer.vocab_size, (args.num_samples, args.answer_len))
    rejected_ids = torch.randint(0, tokenizer.vocab_size, (args.num_samples, args.answer_len))

    prompt_text = tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
    chosen_text = tokenizer.batch_decode(chosen_ids, skip_special_tokens=True)
    rejected_text = tokenizer.batch_decode(rejected_ids, skip_special_tokens=True)

    dataset_dict = {
        "prompt": prompt_text,
        "chosen": chosen_text,
        "rejected": rejected_text,
    }
    
    dataset = Dataset.from_dict(dataset_dict)
    return dataset

parser = argparse.ArgumentParser()
parser.add_argument("--chunk_size", type=int, default=3000)
parser.add_argument("--mode", type=str, default="stream")
parser.add_argument("--prompt_len", type=int, default=1000)
parser.add_argument("--answer_len", type=int, default=8000)
parser.add_argument("--num_samples", type=int, default=50, help="Number of samples to generate")
parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B", help="Model to use for training")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
parser.add_argument("--use_lora", action="store_true", help="Use LoRA for training")
args = parser.parse_args()

lora_config = LoraConfig(
    r=32,
    lora_alpha=128,
    target_modules="all-linear",
    lora_dropout=0.0,
    bias="none",
)

model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
# model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32)

if args.mode == "stream":
    model = StreamModel(model, gradient_accumulation_steps=1, logits_chunk_size=100, stream_checkpoint=True, checkpoint_chunk_size=args.chunk_size)
    TrainerClass = FusedDPOTrainer
elif args.mode == "base":
    TrainerClass = DPOTrainer
else:
    raise ValueError(f"Invalid mode: {args.mode}")

if args.use_lora:
    model = get_peft_model(model, lora_config)

model.gradient_checkpointing_enable()
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
train_dataset = create_dummy_dataset(args)

training_args = DPOConfig(output_dir=args.model_name + "-DPO",
                          logging_steps=10,
                          per_device_train_batch_size=args.batch_size,
                          gradient_accumulation_steps=1,
                          max_prompt_length=None,
                          max_completion_length=None,
                          max_length=None,
                          learning_rate=0., # for gradient profiling
                          )
trainer = TrainerClass(model=model,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    callbacks=[GradientMonitorCallback()])
trainer.train()