#!/usr/bin/env python
"""
Script Name: llava-caption.py
Description: Tool for automatically captionong a directory of text files containing prompts
and corresponding PNG images using the llava multimodal model

Author: David "Zanshinmu" Van de Ven
Date: 4-9-2023
Version: 0.6

As of now, huggingface transformers and llama-cpp-python are working, with
hf transformers falling back to CPU on Apple Silicon due to a LLava bug.
ollama works locally and remotely via OLLAMA_REMOTEHOST
DualModel is the most effective autocaptioner, but requires the most GPU and memory resources
"""
import httpx
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, LlavaNextConfig
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler
from huggingface_hub import hf_hub_download
from PIL import Image
from ollama import Client, pull
from tqdm import tqdm
import pandas as pd
from io import StringIO
import ollama
import subprocess
import torch
import string
import re
import sys
import os
import base64
import tempfile
import ast

''' Here we set the class to use for processing
    Environment variable LLAVA_PROCESSOR can also be used

    The options are currently:
    OLModel  - Process with Ollama application
    HFModel  - Process with Huggingface Transformers
    LCPModel - Process with Llama C++ python bindings
    DualModel - Process with Llama C++ python bindings with Mixtral and Llava collaborating
'''
MODEL = "LCPModel"

# System prompt: You may want to do something different, but we attempt to limit
# model to only the words that appear in the prompt text
SYSTEM_PROMPT = "Describe the elements in the image using the quoted words only:"

# Temperature: Go higher if you want the output to be more creative
TEMPERATURE = 0.6

# Prompt text Preprocessor defaults to off
PREPROCESSOR = bool(ast.literal_eval(os.environ.get('PREPROCESSOR', "False")))
# Secondary caption processing with DualModel, defaults to off
SECONDARY_CAPTION = bool(ast.literal_eval(os.environ.get('SECONDARY_CAPTION', "False")))

# Set to 0 to disable GPU
N_GPU_LAYERS = -1

# System logging, set "True" or "False" here or in env
SYS_LOGGING = bool(ast.literal_eval(os.environ.get('SYS_LOGGING', "False")))
# Logging of processing: set "True" or "False" here or in env
LOGGING = bool(ast.literal_eval(os.environ.get('LOGGING', "False")))

# Ollama RemoteHost: sets the address and port for the Ollama remote host.
OLLAMA_REMOTEHOST = "127.0.0.1:11434"


# Ollama host processor
class OllModel(object):
    def __init__(self, model="llava:7b-v1.5-q4_K_S", olremote="127.0.0.1:11434"):

        self.OLREMOTE = os.environ.get("OLLAMA_REMOTEHOST", olremote)
        # open a client connection
        self.client = Client(host=self.OLREMOTE)
        print(f"\nInitializing Ollama Processor\n")
        self.my_model = model
        try:
            # Check model, if it doesn't exist pull it
            self.client.show(self.my_model)
        except httpx.ConnectError:
            self.ollama_start()
        except ollama.ResponseError as e:
            print('Error:', e.error)
            if e.status_code == 404:
                self.ollama_pull_model(self.my_model)
        else:
            print(f"Connected to Ollama:{self.OLREMOTE}:{self.my_model}\n\n")

    def ollama_start(self):
        process = subprocess.run(['ollama', 'list'])
        if process.returncode != 0:
            print(f"Error calling ollama : {process.stderr}")
            return None
        else:
            return 1

    def ollama_pull_model(self, model):
        print(f"Pulling {model}")
        current_digest, bars = '', {}
        for progress in self.client.pull(model, stream=True):
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
        image_sequence = [image]
        settings = ollama.Options(num_predict=150, temperature=TEMPERATURE)
        response = self.client.generate(self.my_model, instruct, images=image_sequence, options=settings)["response"]
        return response

    def llm_completion(self, system, text, label, json_format=False):
        # First create the instructions
        instruct = f"{system}\n {label}'{text}'"
        settings = ollama.Options(num_predict=1024, seed=31337, temperature=0.1)
        # Format can only be json so set if we want it to be json
        if json_format:
            response = self.client.generate(self.my_model, instruct, options=settings, format="json")["response"]
        else:
            response = self.client.generate(self.my_model, instruct, options=settings)["response"]

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
            do_sample=True,
            temperature=TEMPERATURE,
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
    def __init__(self, repo_id="jartine/llava-v1.5-7B-GGUF", clip_model="llava-v1.5-7b-mmproj-Q4_0.gguf",
                 model="llava-v1.5-7b-Q4_K.gguf"):
        clip_model = hf_hub_download(repo_id=repo_id,
                                     filename=clip_model)
        self.model = hf_hub_download(repo_id=repo_id,
                                     filename=model)
        self.llava = Llama(
            model_path=self.model,
            chat_handler=Llava15ChatHandler(clip_model_path=clip_model, verbose=SYS_LOGGING),
            max_tokens=150,  # response limit
            n_ctx=4096,  # context limit
            n_gpu_layers=N_GPU_LAYERS,
            logits_all=True,  # needed to make llava work
            temperature=TEMPERATURE,
            verbose=SYS_LOGGING
        )
        print(f"\nLlama C++ model:\n")
        print(f"{os.path.basename(self.model)} initialized\n")

        # prep base64 URI for image

    def image_to_base64_data_uri(self, file_path):
        with open(file_path, "rb") as img_file:
            base64_data = base64.b64encode(img_file.read()).decode('utf-8')
        return f"data:image/png;base64,{base64_data}"

        # Process text/images with Llama C++ python bindings

    def llava_completion(self, prompt, image_path):
        image_uri = self.image_to_base64_data_uri(image_path)
        output = self.llava.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are an assistant who describes images exactly as instructed"},
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
        choices = output['choices'][0]
        response = choices['message']['content']
        if LOGGING:
            print(f"\nLLAVA OUTPUT: {response}\n")
        return response

    def process_image(self, text, image_path):
        response = self.llava_completion(text, image_path)
        return response


"""
Two-model processing, Llava 1.5 and Mixtral
Mixtral is running on Ollama, Llava on LLama C++ Python
Mixtral parses the prompt and queries LLava for each element
then Mixtral constructs a caption from the responses.
Why Ollama?  Tried two instances of Llama C++ python
But it seems to not like that.  Ollama works fine
even though it has a llama c++ back-end. 
"""


class DualModel:
    def __init__(self):
        llava_repo = "PsiPi/liuhaotian_llava-v1.5-13b-GGUF"
        # Using a 13B llava to minimize hallucination
        clip_model = "mmproj-model-Q5_0.gguf"
        llava_model = "llava-v1.5-13b-Q5_K_M.gguf"
        self.llava = LCPModel(llava_repo, clip_model, llava_model)
        # Using a known good Mixtral from Ollama
        mixtral_ollama = "mixtral:8x7b-instruct-v0.1-q5_0"
        self.llm = OllModel(mixtral_ollama)

    def strip_text(self, input):
        text = input.translate(str.maketrans('', '', string.digits))
        response = text.translate(str.maketrans('', '', string.punctuation))
        return response

    # Parse out the response from the message
    def parse_response(self, message):
        # Parse the returned dictionary for response
        choices = message['choices'][0]
        response = choices['message']['content']
        return response

    def image_to_base64_data_uri(self, file_path):
        with open(file_path, "rb") as img_file:
            base64_data = base64.b64encode(img_file.read()).decode('utf-8')
        return f"data:image/png;base64,{base64_data}"

    def llm_completion(self, prompt, system, label="text:"):
        # Using our instance of OLModel to invoke the LLM
        # Add system tags if necessary
        system = f"[INST]{system}[/INST]"
        output = self.llm.llm_completion(system, prompt, label)
        return output

    def elements_completion(self, text):
        # Prompt for list of elements from image prompt
        instruction = (f"Follow these instructions consistently to complete the task:\n"
                       f"1. Construct a list of comma-separated elements from the element text.\n"
                       f"2. An element is an object, person, adjective, action or description from the element text.\n"
                       f"3. Do not modify elements. Use identical words as element text. Do not create new elements.\n"
                       f"4. Each property, action, or description of an object or person must be a separate element.\n"
                       f"5. Each element must only be added once. Do not number elements.\n"
                       f"6. Use only the element text. Do not use instructions as elements.\n"
                       f"7. Do not comment, note or explain.  Do not produce any text but the elements.\n"
                       )
        # First we instruct the LLM based on the image prompt text
        list_response = self.llm_completion(instruction, text, "Element text:")
        # Turn the list into a pandas dataframe
        elements = list(list_response.split(','))
        df = pd.DataFrame(elements, columns=['Element'])
        if LOGGING:
            print(f"\nElements:\n{elements}\n")
        return df

    def questions_completion(self, elements):
        # Prompt for list of questions from elements
        q_column = "Question"
        instruction = (f"Follow these instructions to complete the task:\n"
                       f"1. Process the element text into a question.\n"
                       f"2. Example question: 'Is <element> visible?' Use the example as a template.\n"
                       f"3. The question must require a yes/no answer to verify element is in an image\n"
                       f"4. Uwe only the exact element text to create the question..\n"
                       f"5. Do not add instructions. Do not use the words 'text' or 'element' in question.\n "
                       f"6. Respond only with a question. Do not comment, explain or note.\n"
                       )
        print(f"Generating Questions from Elements.\n")
        questions = []
        for index in tqdm(elements.index):
            e = elements.iloc[index]["Element"]
            # Generate question, insert item into new column in dataframe
            response = self.llm_completion(instruction, e, "element text:")
            questions.append(response)
        # Merge the liat with the dataframe
        elements[q_column] = questions
        if LOGGING:
            print(f"Questions: {elements}")
        return elements

    def query_llava_completion(self, questions, image_path):
        # "ask" Llava each question about the elements of the image
        llava_response = ""
        print(f"Querying Llava model with questions.\n")
        for index in tqdm(questions.index):
            question = str(questions.iloc[index]['Question'])
            element = str(questions.iloc[index]['Element'])
            # asking Llava about the element
            answer = self.llava.llava_completion(f"Answer accurately with yes/no: {question}:{element}?", image_path)
            if LOGGING:
                print(f"Results: {index}\n {question}\n {element}\n {answer}\n")
            if "Yes" in answer:
                # Add this to list of valid tokens for caption
                llava_response += f"{element}, "

        if LOGGING:
            print(f"Visible: {llava_response}\n")
        return llava_response

    def caption_completion(self, visible):
        instruction = (f"Follow these instructions to complete the task:\n"
                       f"1. Process the text into elements for an image caption.\n"
                       f"2. Organize the elements of the text into a logical structure for image description.\n"
                       f"3. Put the subject of the image first with related description.\n"
                       f"4. Background elements and elements not related to the subject come later.\n"
                       f"5. Do not modify the text or add anything, just change the structure."
                       f"6. Do not leave any of the text elements out or modify them."
                       f"7. If the subject is a person, use appropriate pronouns and nouns."
                       f"8. Make sure the list is comma-separated. No punctuation or numbers should be used.\n"

                       )
        response = self.llm_completion(instruction, visible)
        if LOGGING:
            print(f"Caption: {response}\n")

        return response

    # Here is the heart of the DualModel Magic
    def process_image(self, prompt, image_path):
        # Create list of elements
        elements = self.elements_completion(prompt)
        # Create list of questions from elements
        questions = self.questions_completion(elements)
        # Process list of questions with LLava model
        visible = self.query_llava_completion(questions, image_path)
        if SECONDARY_CAPTION:
            # Now, synthesize the visible elements into a caption
            caption = self.caption_completion(visible)
            return caption
        else:
            # Just return Visible which is minimalist but accurate
            return visible


# prep prompt text for processing
# regex, format re.sub('<text to find>','<text to replace>')
def preprocess(text):
    if not PREPROCESSOR:
        return text
    pattern = r'(photo of\s\w+),'
    new_text = re.sub(pattern, 'photo of', text)
    new_text = re.sub('Cybergirl', 'woman', new_text)
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
        # Let user know we're doing something as this can take a while
        print(f"Processing {png_filename}")
        # Open and resize the PNG image
        original = Image.open(png_filename)
        img = resize_image(original)
        # Save resized PNG so we have a path
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        img.save(temp_file)
        original.close()
        return temp_file.name
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
                    text = preprocess(f.read())
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
        processor_name = str(newclass)
        processor_name = processor_name[processor_name.index('.') + 1:]
        print(f"\n<'{processor_name} processor loading\n")
        # Instantiate the processor
        ModelClass = newclass()

    directory = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    main(directory)
