# llava-caption.py
## Automatically captions a directory of text files containing prompts
## and corresponding PNG images using the llava multimodal model

# Installation Guide

This guide provides instructions for setting up the necessary requirements for `llava-caption.py`

---

## Setting Up Python Virtual Environment and Installing Dependencies

This repository utilizes Python 3 and relies on specific dependencies to function properly. To ensure a consistent environment and manage these dependencies efficiently, it's recommended to set up a Python virtual environment. This isolates the project's dependencies from other Python projects on your system.

### Setting Up a Python Virtual Environment

1. **Install Python 3**: If you haven't already, [download and install Python 3](https://www.python.org/downloads/) for your operating system.

2. **Install `virtualenv` (if not already installed)**: `virtualenv` is a tool used to create isolated Python environments. If you haven't installed it yet, you can do so via pip, Python's package installer. Run the following command in your terminal:

    ```
    pip install virtualenv
    ```

3. **Create a Virtual Environment**: Navigate to your project directory in the terminal and create a new virtual environment by running:

    ```
    python3 -m venv venv
    ```

    This command will create a folder named `venv` in your project directory, containing the Python interpreter and standard library for your virtual environment.

4. **Activate the Virtual Environment**: Before you can install dependencies or run your project within the virtual environment, you need to activate it. On macOS/Linux, run:

    ```
    source venv/bin/activate
    ```

    On Windows, run:

    ```
    venv\Scripts\activate
    ```

    Once activated, you should see `(venv)` prefixed to your terminal prompt, indicating that you are now working within the virtual environment.

### Installing Dependencies

This project uses a `requirements.txt` file to specify its dependencies. To install these dependencies, ensure that your virtual environment is activated, and then run:

```
pip install -r requirements.txt
```

This command will install all the required dependencies listed in the `requirements.txt` file.

### Deactivating the Virtual Environment

Once you're done working on your project, you can deactivate the virtual environment by simply running:

```
deactivate
```

This will return you to your system's default Python environment.

By following these steps, you'll have a clean, isolated environment for your Python project, with all the necessary dependencies installed.

## Setting `TORCH_DEVICE` Environment Variable for PyTorch

When working with PyTorch, it's often necessary to specify the device on which tensors are allocated. This can be particularly important when dealing with GPUs for acceleration. PyTorch provides the `torch.device` class to handle this.

By setting the `TORCH_DEVICE` environment variable, you can conveniently specify the default device for PyTorch operations without having to modify your script each time.

## Setting up `TORCH_DEVICE`

Follow these steps to set up the `TORCH_DEVICE` environment variable:

### 1. Determine the Device

Decide on the device you want to use. This could be a CPU or a specific GPU.

### 2. Set `TORCH_DEVICE`

Set the `TORCH_DEVICE` environment variable to specify the desired device. You can do this in your terminal or shell script.

For example, to set it to use GPU 0:

```bash
export TORCH_DEVICE="cuda:0"
```
## Setting `LLAVA_PROCESSOR` Environment Variable

The script allows you to select the Llava model processor with an environment variable which will be used to load the appropriate class to process text and images. 

The current options are:
    OLModel  - Process with Ollama application
    HFModel  - Process with Huggingface Transformers
    LCPModel - Process with Llama C++ python bindings

## Setting up `LLAVA_PROCESSOR`

Follow these steps to set up the `LLAVA_PROCESSOR` environment variable:

### 1. Determine the processor

Decide on the processor you want to use. 

### 2. Set `LLAVA_PROCESSOR`

Set the `LLAVA_PROCESSOR` environment variable to specify the desired processor. You can do this in your terminal or shell script.

For example, to set it to use Ollama:

```bash
export LLAVA_PROCESSOR="OLModel"
```

# Caveats

- The appropriate LLava models will be automatically downloaded using huggingface-hub, or with Ollama.
    This can take some time. 
- OLModel assumes you have Ollama installed locally, but will eventually support remote Ollama hosts
- The current HFModel implementation does not use MPS on Apple Silicon due to a bug in LLava
- HFModel defaults to CPU.  See above for how to specify your device with env variables. 
- The memory requirements are high and it takes a lot of CPU to run the model with HFModel in CPU mode
- The script assumes you have already preproccessed the PNG files in the target directory to extract the prompts to text files with the same names. 
- The script will overwrite the text files with the generated captions

## Usage

The `llava-caption.py` script can be used as follows:

- Process a directory of images and prompt files:
  ```bash
  python3 llava-caption.py /path/to/image/folder/
  ```
