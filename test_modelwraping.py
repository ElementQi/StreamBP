from fused_backward_model import StreamModel
from transformers import AutoModelForCausalLM
import torch
import time

torch.set_printoptions(precision=8)

def clean_grad(model):
    for param in model.parameters():
        param.grad = None

# USE_STREAM = True
# USE_MINIS = True
# MODE = "minis"
MODE = ""
RECORD_MEMORY = False
gradient_accumulation_steps = 1

# load data
input_ids = torch.load("test_data_model/input_ids.pt")
attention_mask = torch.load("test_data_model/attention_mask.pt")
labels = torch.load("test_data_model/labels.pt")

batch_size = 1
input_ids = torch.cat([input_ids for _ in range(batch_size)], dim=0)
attention_mask = torch.cat([attention_mask for _ in range(batch_size)], dim=0)
labels = torch.cat([labels for _ in range(batch_size)], dim=0)

seq_expansion = 1
input_ids = torch.cat([input_ids for _ in range(seq_expansion)], dim=1)
attention_mask = torch.cat([attention_mask for _ in range(seq_expansion)], dim=1)
labels = torch.cat([labels for _ in range(seq_expansion)], dim=1)

# base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B").bfloat16().to(input_ids.device)
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B").to(input_ids.device)
base_model.train()

forward_model = base_model
if MODE == "stream":
    print("using stream model")
    # forward_model = StreamModel(base_model, logits_chunk_size=3000, checkpoint_chunk_size=3000*seq_expansion)
    forward_model = StreamModel(base_model, logits_chunk_size=100, checkpoint_chunk_size=1000, stream_checkpoint=True)
    forward_model.gradient_checkpointing_enable()
elif MODE == "minis":
    from minis.mini_sequence import minisequence
    forward_model = minisequence(base_model, chunk_size=1000)
else:
    print("using base model with gradient checkpointing")
    base_model.gradient_checkpointing_enable()

if RECORD_MEMORY:
    torch.cuda.memory._record_memory_history(max_entries=1000000)

# forward_model.gradient_checkpointing_enable()
t1 = time.time()

for i in range(gradient_accumulation_steps):
    output = forward_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True
    )

    if output.loss.requires_grad:
        output.loss.backward()
    
    # clean_grad(forward_model)
    
if RECORD_MEMORY:
    torch.cuda.memory._dump_snapshot(f"test_data_model/memory_record_modelwrap.pickle")
    torch.cuda.memory._record_memory_history(enabled=None)

print("Time taken: ", time.time() - t1)
print("allocated: ", torch.cuda.memory_allocated() / 1e9, "max allocated: ", torch.cuda.max_memory_allocated() / 1e9)

print(output.loss)
# print(forward_model.lm_head.weight.grad[0])
# print(forward_model.model.model.layers[0].self_attn.q_proj.weight.grad[0])
# print(forward_model.model.layers[0].self_attn.q_proj.weight.grad[0])