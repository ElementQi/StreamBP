Add 2 lines of code to save over 10 GB memory, without introducing additional error or time overhead.

### Usage
Add 3 lines in the initialization of your `Trainer`
```python
from fused_backward_model import FusedBackwardModel

class YourTrainerClass(Trainer):
    def __init__(
        self, *args, **kwargs
    ) -> None:
        kwargs["model"] = FusedBackwardModel(kwargs["model"], chunk_size=200) # wrap the 

        super().__init__(**kwargs)
        self.accelerator.backward = lambda loss: None # backward is fused with forward, no need to call accelerator.backward
        
        ... # your code
```