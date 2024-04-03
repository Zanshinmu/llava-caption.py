#!/usr/bin/env python
"""
Script Name: llava-caption.py
Description: Tool for automatically captionong a directory of text files containing prompts
and corresponding PNG images using the llava multimodal model

Author: David "Zanshinmu" Van de Ven
Date: 4-3-2023
Version: 0.3

As of now, huggingface transformers and llama-cpp-python are working, with
hf transformers falling back to CPU on Apple Silicon due to a LLava bug.
ollama works locally but remote hasn't been implemented
"""

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, LlavaNextConfig
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler
from huggingface_hub import hf_hub_download
from PIL import Image
from ollama import Client, pull
from httpx import RemoteProtocolError, ConnectError
import ollama
import torch
import re
import sys
import os
import base64
import tempfile
import tqdm

# System prompt: You may want to do something different, but we attempt to limit
# model to only the words that appear in the prompt text
SYSTEM_PROMPT = "Describe with no embellishment using the quoted words only the elements in the image:"

# Temperature: Go higher if you want the output to be more creative
TEMPERATURE = 0.6

# Prompt Preprocessor defaults to off
PREPROCESSOR = False

''' Here we set the class to use for processing
    Environment variable LLAVA_PROCESSOR can also be used

    The options are currently:
    OLModel  - Process with Ollama application
    HFModel  - Process with Huggingface Transformers
    LCPModel - Process with Llama C++ python bindings
'''
MODEL = "LCPModel"

# Logging of models
LOGGING = False

# Ollama host processor
class OLModel:
    def __init__(self):
        self.OLMODEL="llava:7b-v1.5-q4_K_S"
        self.client=Client()
        # Check model, if it doesn't exist pull it
        try:
            ollama.show(self.OLMODEL)
        except ollama._types.ResponseError:
            self.ollama_pull_model(self.OLMODEL)
        except RemoteProtocolError:
            print ("Lost connection to Ollama")
        except ConnectError:
            print ("Connection Refused")
        else:
            print (f"Connected to Ollama: {self.OLMODEL}")
    
    def ollama_pull_model(self, model):
        print (f"Pulling {model}")
        current_digest, bars = '', {}
        for progress in pull(model, stream=True):
            digest = progress.get('digest', '')
            if digest != current_digest and current_digest in bars:
                bars[current_digest].close()
            if not digest:
                print(progress.get('status'))
            continue
            if digest not in bars and (total := progress.get('total')):
                bars[digest] = tqdm(total=total, desc=f'pulling {digest[7:19]}', unit='B', unit_scale=True)
            if completed := progress.get('completed'):
                bars[digest].update(completed - bars[digest].n)
                current_digest = digest
            
        # Process text/images with Ollama host
    def process_image(self, text, image):
        instruct = f"{SYSTEM_PROMPT}'{text}'"
        image_sequence = []
        image_sequence.append(image)
        settings = ollama.Options(num_predict=150,temperature = TEMPERATURE)
        response = self.client.generate(self.OLMODEL, instruct, images=image_sequence, options = settings)["response"]
        return response
    

# HF Transformer based processing, this is probably best option for CUDA or CPU
# Using Llava-Next (1.6) here, unquantized so it takes some RAM
class HFModel:
    def __init__(self):
        # Huggingface Llava model path
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
            do_sample = True,
            temperature = TEMPERATURE,
        )
        self.processor = LlavaNextProcessor.from_pretrained(HF_MODEL_PATH)
        self.processor.tokenizer.padding_side = "left"
        print(f"{HF_MODEL_PATH} initialized for device '{self.device}'\n")
    
    
    # We don't want a a response cluttered with [INST].
    def striptext(self, text):
        try:
            stripped_text = text.split('[/INST]')[1]
        except:
            return text
        return stripped_text
        
        # Process text/images with HuggingFace transformers library and hopefully MPS/CUDA
    def process_image(self, text, image_path):
        image = Image.open(image_path)
        # Prepare the prompt
        prompt = f"[INST] <{image}>\n{SYSTEM_PROMPT}'{text}'[/INST]"

        # process the prompt
        inputs = self.processor(prompt, image, return_tensors="pt", padding=True).to(self.device)

        # Autoregressively complete the prompt
        self.model.config.image_token_index = 1
        output = self.model.generate(**inputs, max_new_tokens=self.max_tokens,
                                        pad_token_id=self.processor.tokenizer.pad_token_id)

        # Decode the output
        response = self.striptext(self.processor.decode(output[0], skip_special_tokens=True))
        image.close()
        return response


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
        
        
        # prep base64 URI for image
    def image_to_base64_data_uri(self, file_path):
        with open(file_path, "rb") as img_file:
            base64_data = base64.b64encode(img_file.read()).decode('utf-8')
        return f"data:image/png;base64,{base64_data}"

        
        # Process text/images with Llama C++ python bindings
    def process_image(self, text, image_path):
        image_uri = self.image_to_base64_data_uri(image_path)
        prompt = f"{SYSTEM_PROMPT} '{text}'"
        output = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are an assistant who describes images without embellishment as instructed"},
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


# prep prompt text for processing
# regex, format re.sub('<text to find>','<text to replace>')
def preprocess(text):
    new_text = re.sub('Cybergirl', 'woman', text)
    new_text = re.sub('Cyberpunk man', 'man', new_text)
    new_text = re.sub(', photograph, film, professional, highly detailed', '', new_text)
    return new_text


    # image must be resized for model
def resize_image(image):
    # Within model limits and preserves aspect ratio
    # note we are using thumbnail here because it instantiates a new resized image inline without modifying the original
    max_size = (670, 670)

    image.thumbnail(max_size, Image.LANCZOS)
    return image


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
        original.close()
        return  temp_file.name
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
                    image_path = read_corresponding_png(filepath)
                    if image_path:
                        response = ModelClass.process_image(text, image_path)
                        processed += 1
                        with open(filepath, 'w') as f:
                            f.write(response)
                            print(f"{file}: {processed} of {filecount}\n {response}\n")
                    else:
                        print(f"No corresponding image for {file}\n")


if __name__ == "__main__":
    # instantiate processing class
    # try environment, then variable
    try:
        newclass = globals()[os.environ.get('LLAVA_PROCESSOR', MODEL)]
    except:
        sys.exit("\nError: Named processor class does not exist!\n")
    else:
        ModelClass = newclass()

    directory = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    main(directory)
