# llava-caption.py
## Automatically captions a directory of text files containing prompts
## and corresponding PNG images using the llava multimodal model
___
Use case: You have a large number of images you intend to use for a training set, but the images were rendered with generative AI so many of the features in the prompts are missing from the images, which makes for bad training.  Manual captioning is too time-consuming.  Enter llava-caption.py which has higher quality than BLIP with the basic processsors, and near-manual quality with the DualModel processor.
___

## DualModel *experimental*
Two-model processing, Llava 1.5 and Mixtral. Mixtral is running on Ollama, Llava on LLama C++ Python
Mixtral parses the prompt and queries LLava for each element then Mixtral constructs a caption from the responses.
Why Ollama?  Two instances of Llama C++ python do not work.  Ollama works fine even though it has a llama c++ back-end.
DualModel is being released as experimental: it is slow and has a tendency to go off-course over time,
but that may be fixable with grammar and optimizations. 

# Installation Guide

This guide provides instructions for setting up the necessary requirements for `llava-caption.py`

---
## Installing Ollama 
Ollama can be downloaded and installed for various architectures and operating systems 
**https://ollama.com/download
___

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
    DualModel - Process with dual models, mixtral with ollama, llava with Llama C++ python

## Setting `LLAVA_PROCESSOR` Environment Variable

Follow these steps to set up the `LLAVA_PROCESSOR` environment variable:

### 1. Determine the processor

Decide on the processor you want to use. 

### 2. Set `LLAVA_PROCESSOR`

Set the `LLAVA_PROCESSOR` environment variable to specify the desired processor. You can do this in your terminal or shell script.

For example, to set it to use Ollama:

```bash
export LLAVA_PROCESSOR="OLModel"
```
## Setting `OLLAMA_REMOTEHOST` Environment Variable

The script allows you to select the IP address of the Ollama host.  Default is localhost. 

## Setting `OLLAMA_REMOTEHOST` Environment Variable

Follow these steps to set up the `OLLAMA_REMOTEHOST` environment variable:

### 1. Determine the address of the host

Decide on the host running Ollama you want to use. 

### 2. Set `OLLAMA_REMOTEHOST`

Set the `OLLAMA_REMOTEHOST` environment variable to specify the desired processor. You can do this in your terminal or shell script.

For example, to set it to use Ollama on 192.168.1.118:

```bash
export OLLAMA_REMOTEHOST="192.168.1.118:11434"
```

## Setting `SYSTEM_LOGGING` Environment Variable

The SYSTEM_LOGGING environment variable enables console messages from the models which you may need for debugging on your system. 

Follow these steps to set up the `SYSTEM_LOGGING` environment variable:

### 2. Set `SYSTEM_LOGGING`

Set the `SYSTEM_LOGGING` environment variable to 'True'. You can do this in your terminal or shell script.

For example:

```bash
export SYSTEM_LOGGING="True"
```
## Setting `LOGGING` Environment Variable

The LOGGING environment variable enables console messages from the script which you may need for debugging on your system. 

Follow these steps to set up the `LOGGING` environment variable:

### 2. Set `LOGGING`

Set the `LOGGING` environment variable to 'True'. You can do this in your terminal or shell script.

For example:

```bash
export LOGGING="True"
```


# Caveats

- The appropriate LLava models will be automatically downloaded using huggingface-hub, or with Ollama.
    This can take some time and disk space.  The models in the script have been carefully selected for performance and resource use.
- OLModel assumes you have Ollama installed locally by default but can use a remote host via OLLAMA_REMOTEHOST
- The current HFModel implementation does not use MPS on Apple Silicon due to a bug in LLava 1.6
- HFModel defaults to CPU.  See above for how to specify your device with env variables. 
- The memory requirements are high and it takes a lot of CPU to run the model with HFModel in CPU mode
- The script assumes you have already preproccessed the PNG files in the target directory to extract the prompts to text files with the same names. 
- The script will overwrite the prompt text files in the target directory with the generated captions
- DualModel is the most accurate but requires at least 64GB of Unified Memory on a Mac, and as much GPU as you can give it. 
  However, it is possible to run the LLava model locally and the Mixtral model on a remote host via Ollama. (see OLLAMA_REMOTEHOST)

## Usage

The `llava-caption.py` script can be used as follows:

- Process a directory of images and prompt files:
  ```bash
  python3 llava-caption.py /path/to/image/folder/
  ```
