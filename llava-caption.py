#!/usr/bin/env python
"""
Script Name: llava-caption.py
Description: Tool for automatically captionong a directory of text files containing prompts
and corresponding PNG images using the llava multimodal model

Author: David "Zanshinmu" Van de Ven
Date: 4-2-2023
Version: 0.2

As of now, huggingface transformers and llama-cpp-python are working, with
hf transformers falling back to CPU on Apple Silicon due to a LLava bug.
"""

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, LlavaNextConfig
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler
from huggingface_hub import hf_hub_download
from PIL import Image
import torch
import re
import sys
import os
import base64
import tempfile

# System prompt: You may want to do something different, but we attempt to limit
# model to only the words that appear in the prompt text
SYSTEM_PROMPT = "Efficiently describe image using only these words if detected:"
# Temperature: Go higher if you want the output to be more creative
TEMPERATURE = 0.6
# Prompt Preprocessor defaults to off
PREPROCESSOR = False
# For now, using switches to determine how we process the images and text
HFPROCESSING = False
LCPPROCESSING = True
# Logging of models
LOGGING = False

# HF Transformer based processing, this is probably best option for CUDA or CPU
# Using Llava-Next (1.6) here, unquantized so it takes some RAM
class HFModel:
    def __init__(self):
        # Set up hf transformers
        HF_MODEL_PATH = "llava-hf/llava-v1.6-mistral-7b-hf"
        # set torch device from env, default is cpu
        self.device = torch.device(os.environ.get('TORCH_DEVICE', "cpu"))
        self.max_tokens = os.environ.get('MAX_TOKENS', 150)
        if self.device.type == "mps":
            self.dtype = torch.float16
        elif self.device.type == "cpu":
            self.dtype = torch.float32
        elif self.device.type == "cuda":
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32
            
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            HF_MODEL_PATH, torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            device_map=self.device,
            temperature = TEMPERATURE,
        )
        self.processor = LlavaNextProcessor.from_pretrained(HF_MODEL_PATH)
        self.processor.tokenizer.padding_side = "left"
        print(f"{HF_MODEL_PATH} initialized for device '{self.device}'\n")

# LLama C++ based processing, Llava 1.5 works best here
# Using quantized model which should work on 16GB Apple Silicon
class LCPModel:
    def __init__(self):
        self.repo = "jartine/llava-v1.5-7B-GGUF"
        clip_model = hf_hub_download(repo_id=self.repo,
                                          filename="llava-v1.5-7b-mmproj-Q4_0.gguf")
        self.model = hf_hub_download(repo_id=self.repo,
                                     filename="llava-v1.5-7b-Q4_K.gguf")
        self.llm = Llama(
            model_path=self.model,
            chat_handler=Llava15ChatHandler(clip_model_path=clip_model,verbose=False),
            max_tokens=150,  # response limit
            n_ctx=4096,  # context limit
            n_gpu_layers=1,
            logits_all=True,  # needed to make llava work
            temperature = TEMPERATURE,
            verbose=LOGGING
        )
        print(f"\nLLama C++ model {os.path.basename(self.model)} initialized\n")


# We don't want a cluttered response.
def striptext(text):
    try:
        stripped_text = text.split('[/INST]')[1]
    except:
        return text
    return stripped_text


# prep prompt text for processing
# regex, format re.sub('<text to find>','<text to replace>')
def preprocess(text):
    new_text = re.sub('Cybergirl', 'woman', text)
    new_text = re.sub('Cyberpunk man', 'man', new_text)
    new_text = re.sub(', photograph, film, professional, highly detailed', '', new_text)
    return new_text


# prep base64 URI for image
def image_to_base64_data_uri(file_path):
    with open(file_path, "rb") as img_file:
        base64_data = base64.b64encode(img_file.read()).decode('utf-8')
    return f"data:image/png;base64,{base64_data}"


def resize_image(image):
    # image must be resized for model
    # Within model limits and preserves aspect ratio
    # note we are using thumbnail here because it instantiates a new resized image inline without modifying the original
    max_size = (670, 670)

    image.thumbnail(max_size, Image.LANCZOS)
    return image


# Process text/images with HuggingFace transformers library and hopefully MPS/CUDA
def process_image_hf(text, image):
    # Prepare the prompt
    prompt = f"[INST] <{image}>\n{SYSTEM_PROMPT}'{text}'[/INST]"

    # process the prompt
    inputs = hfmodel.processor(prompt, image, return_tensors="pt", padding=True).to(hfmodel.device)

    # Autoregressively complete the prompt
    hfmodel.model.config.image_token_index = 1
    output = hfmodel.model.generate(**inputs, max_new_tokens=hfmodel.max_tokens,
                                    pad_token_id=hfmodel.processor.tokenizer.pad_token_id)

    # Decode the output
    response = striptext(hfmodel.processor.decode(output[0], skip_special_tokens=True))
    return response

# Process text/images with Llama C++ python bindings
def process_image_lcp(text, image_path):
    image_uri = image_to_base64_data_uri(image_path)
    prompt = f"{SYSTEM_PROMPT} '{text}'"
    output = lcpmodel.llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are an assistant who efficiently describes images, without embellishment."},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_uri}},
                    {"type": "text", "text": {prompt}}
                ]
            }
        ]
    )
    # Parse the returned dictionary for response
    choices=output['choices'][0]
    response = choices['message']['content']
    return response


# Find the corresponding PNG file to the text file
def read_corresponding_png(text_file):
    # Get the filename without extension
    filename_without_extension = os.path.splitext(text_file)[0]

    # Check if the corresponding PNG file exists
    png_filename = filename_without_extension + ".png"
    if os.path.exists(png_filename):
        # Open and resize the PNG image
        original = Image.open(png_filename)
        img = resize_image(original)
        # Save resized PNG so we have a path
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        img.save(temp_file)
        llavaimage = Image.open(temp_file)
        original.close()
        return llavaimage, temp_file.name
        # print(f"PNG image '{png_filename}' opened successfully.")
    else:
        return None


def count_text_files(file_list):
    return sum(1 for file_path in file_list if file_path.endswith('.txt'))


# Walk the image dirs
def main(directory):
    for root, dirs, files in os.walk(directory):
        filecount = count_text_files(files)
        processed = 0
        for file in files:
            if file.endswith('.txt'):
                filepath = os.path.join(root, file)
                with open(filepath, 'r') as f:
                    if PREPROCESSOR:
                        text = preprocess(f.read())
                    else:
                        text = f.read()
                    image,image_path = read_corresponding_png(filepath)
                    if image:
                        if HFPROCESSING:
                            response = process_image_hf(text, image)
                        if LCPPROCESSING:
                            response = process_image_lcp(text, image_path)
                        image.close()
                        processed += 1
                        with open(filepath, 'w') as f:
                            f.write(response)
                            print(f"{file}: {processed} of {filecount}\n {response}\n")
                    else:
                        print(f"No corresponding image for {file}\n")


if __name__ == "__main__":
    if HFPROCESSING:
        hfmodel = HFModel()

    if LCPPROCESSING:
        lcpmodel = LCPModel()

    directory = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    main(directory)
