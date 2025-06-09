# Pytest Tutorial

## For testing BP correctness and peak memory usage

1. Install pytest (if you want to use this feature)

    ```bash
    pip install pytest
    ```

2. Activate your conda environment (replace with your environment name)

    ```bash
    conda activate streambp
    ```

3. Navigate to StreamBP directory

    ```bash
    cd /path/to/your/StreamBP
    ```

4. Set up CUDA configuration

    ```bash
    # optional, most of the time reduces peak memory usage
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
    export CUDA_VISIBLE_DEVICES=0
    ```

5. Run tests

    ```bash
    # change the model name inside this file to test different models
    pytest tests/test_grad_correctness.py -s -v
    ```
