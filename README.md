# llava-caption.py
## Automatically captions a directory of text files containing prompts
## and corresponding PNG images using the llava 1.6 model

# Installation Guide

This guide provides instructions for setting up the necessary requirements for `llava-caption.py`

Sure, here's a description you can use for your GitHub README.md:

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

# Caveats

- The current huggingface implementation does not currently use MPS due to a bug
- The memory requirements are quite high and it takes a lot of CPU time to run the model  
- The script assumes you have already preproccessed the PNG files in the target directory to extract the prompts to text files with the same names. 
- The script will overwrite the text files with the generated captions

## Usage

The `llava-caption.py` script can be used as follows:

- Process a directory of images and prompt files:
  ```bash
  python3 llava-caption.py /path/to/image/folder/
  ```
