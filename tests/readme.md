# Pytest Tutorial

## For testing BP correctness and peak memory usage

1. Activate your conda environment (replace with your environment name)

    ```bash
    conda activate ${CONDA_ENV:-streambp}
    ```

2. Navigate to StreamBP directory

    ```bash
    cd /path/to/your/StreamBP
    ```

3. Set up Python path and CUDA configuration

    ```bash
    export PYTHONPATH=.
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export CUDA_VISIBLE_DEVICES=${GPU_DEVICE:-0}
    ```

4. Run tests

    ```bash
    # change the model name inside this file
    pytest tests/test_grad_correctness.py -s -v
    ```
