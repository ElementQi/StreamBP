from ..stream_model import StreamModel
from transformers import Trainer

class StreamSFTTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

        # TODO: This is too hacky, find a better way to do this
        self.accelerator.backward = lambda loss: None # backward is fused with forward, no need to call accelerator.backward