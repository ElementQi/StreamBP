from fused_backward_model import StreamModel
from transformers import Trainer

class FusedSFTTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        # kwargs["model"] = FusedBackwardModel(kwargs["model"], chunk_size=200) # wrap the model with FusedBackwardModel
        # assert isinstance(kwargs["model"], StreamModel), "model must be a StreamModel"
        super().__init__(**kwargs)
        self.accelerator.backward = lambda loss: None # backward is fused with forward, no need to call accelerator.backward
        