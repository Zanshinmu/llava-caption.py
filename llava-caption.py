#!/usr/bin/env python
"""
Script Name: llava-caption.py
Description: Tool for automatically captioning a directory of text files containing prompts
and corresponding PNG images using the llava multimodal model

Author: David "Zanshinmu" Van de Ven
Date:11-11-2024
Version: 0.70

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
from ollama import Client, pull, generate
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import json
import json_repair
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
MODEL = "VisionModel"

# System prompt: You may want to do something different, but we attempt to limit
# model to only the words that appear in the prompt text
SYSTEM_PROMPT = "Describe the image following this style:"

# Temperature: Go higher if you want the output to be more creative
TEMPERATURE = 0.0

# Prompt text Preprocessor defaults to off
PREPROCESSOR = bool(ast.literal_eval(os.environ.get('PREPROCESSOR', "True")))
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
        settings = ollama.Options(num_predict=1024, seed=31337, temperature=0.0)
        # Ollama format can only be json so set if we want it to be json
        json_header = "Respond only in JSON with the response string named 'response':"
        if json_format:
            instruct = f"{json_header}{system}\n {label}'{text}'"
            response = self.client.generate(self.my_model, instruct, options=settings, format="json")["response"]
        else:
            instruct = f"{system}\n {label}'{text}'"
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
        clip_model = "mmproj-model-Q4_0.gguf"
        llava_model = "llava-v1.5-13b-Q4_0.gguf"
        self.llava = LCPModel(llava_repo, clip_model, llava_model)
        # Using a known good Mixtral from Ollama
        mixtral_ollama = "mixtral:8x7b-instruct-v0.1-q5_0"
        self.llm = OllModel(mixtral_ollama)

    def strip_text(self, input):
        # Remove newlines
        stripped = input.replace("\n", "")
        text = stripped.translate(str.maketrans('', '', string.digits))
        response = text.translate(str.maketrans('', '', string.punctuation))
        return response

    # Parse out the response from the message
    def parse_response(self, message):
        # Parse the returned dictionary for response
        choices = message['choices'][0]
        response = choices['message']['content']
        return response
        
    def identify_subject(self, e,  context):
        instruction = (f"Follow instructions and respond in json.\n"
                       f"1. Find the owner of '{e}' in the context text. Is it a man, woman, object or background of the image?\n"
                       f"2. Use 2 words or less from the context text to identify the owner of {e}. \n"
                       f"3. Respond with only the requested result. Return only two words in json response.\n"
                       )
        s = self.llm_completion(instruction, context, "context:", json_format=True)
        j = json_repair.loads(s)
        for key in j:
            if key != 'response':
                if LOGGING:
                    print (j, key)
            else:
                subject=j[key]
                if LOGGING:
                    print (f"Subject of '{e}' is {subject}\n")
                
        return subject
        
    def image_to_base64_data_uri(self, file_path):
        with open(file_path, "rb") as img_file:
            base64_data = base64.b64encode(img_file.read()).decode('utf-8')
        return f"data:image/png;base64,{base64_data}"

    def llm_completion(self, prompt, system, label="text:", json_format=False):
        # Using our instance of OLModel to invoke the LLM
        # Add system tags if necessary
        system = f"[INST]{system}[/INST]"
        if json_format:
            output = self.llm.llm_completion(system, prompt, label, json_format='json')
        else:
            output = self.llm.llm_completion(system, prompt, label)

        return output

    def elements_completion(self, text):
        # Prompt for list of elements from image prompt
        instruction = (f"Follow these instructions to complete the task:\n"
                       f"1. Compile a list of comma-separated elements from the element text.\n"
                       f"2. Organize the elements by subject and description or action of that subject. \n"
                       f"3. Use identical words to element text. Use 'man' or 'woman' instead of a name.\n"
                       f"4. Each property, action, or description of a subject must be a separate element.\n"
                       f"5. Each element must only be added once. Do not number the list.\n"
                       f"7. Respond only with the comma-separated list of elements.\n"
                       )
        # First we instruct the LLM based on the image prompt text
        list_response = self.llm_completion(instruction, text, "Element text:")
        # Turn the list into a pandas dataframe
        clean_text = list_response.replace("\n", "")
        elements = list(clean_text.split(','))
        df = pd.DataFrame(elements, columns=['Element'])
        if LOGGING:
            print(f"\nElements:\n{elements}\n")
        return df

    def questions_completion(self, elements):
        # First, prepare context to work from
        context = ', '.join(elements['Element'])
        # Prompt for list of questions from elements
        q_column = "Question"
        instruction = (f"Perform the task without remarks.\n"
                       f"Adhere strictly to these guidelines:\n"
                       f"1. Process the element text into a simple, direct question relating to the subject.\n"
                       f"2. Example question: 'Is <subject> <element text> in the image?'\n"
                       f"3. The question must require a yes/no answer to verify element is visible in an image\n"
                       f"4. Use exact element text to create the question. Try to use correct grammar.\n"
                       f"5. Use the subject and the element text for the question.\n "
                       f"6. Respond only with a simple question.\n"
                       )
        print(f"Generating Questions from Elements.\n")
        questions = []
        for index in tqdm(elements.index):
            e = elements.iloc[index]["Element"]
            subject = self.identify_subject(e, context)
            # Generate question, insert item into new column in dataframe
            # Using JSON to enforce
            response = self.llm_completion(instruction, e, f"subject: {subject} element text:", json_format=True)
            j = json_repair.loads(response)
            for key in j:
                if key != 'response':
                    if LOGGING:
                        print (j, key)
                else:
                    questions.append(j[key])
                        
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
                       f"8. Make sure the list is comma-separated and not numbered.\n"

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
        # sanitize the string
        visible = self.strip_text(visible)
        if SECONDARY_CAPTION:
            # Now, synthesize the visible elements into a caption
            caption = self.caption_completion(visible)
            return caption
        else:
            # Just return Visible which is minimalist but accurate
            return visible

"""
Single model processing, Llama 3.2 Vision
The model parses the prompt and queries the image for each element
then constructs a natural language caption from the responses.
The output of this one is intended for FLux/SD3 models.
Uses Ollama for processing. 
"""


class VisionModel:
    def __init__(self):
        # Llama 3.2 vision instruct
        self.model_ollama = "llama3.2-vision:11b-instruct-q8_0"
        self.options = {'temperature': 0, 'num_predict':160}
        self.llm = OllModel(self.model_ollama)


    def strip_text(self, input):
        # Remove newlines
        stripped = input.replace("\n", "")
        
        # Remove digits
        text = stripped.translate(str.maketrans('', '', string.digits))
        
        # Custom punctuation string excluding hyphen (-), period (.), and comma (,)
        custom_punctuation = string.punctuation.replace("-", "").replace(".", "").replace(",", "")
        
        # Remove all punctuation except for hyphens, periods, and commas
        response = text.translate(str.maketrans('', '', custom_punctuation))
        
        return response

    def llm_completion(self, prompt, image_path, format='json', system = "You are an image captioning assistant who accurately and concisely describes the visual elements present in an image, including objects, colors, and spatial relationships, focusing only on what is visible, without embellishment or metaphor. You never refer to the image directly, you only describe the contents. You do not interpret the image, you describe it objectively." ):
        image = base64.b64encode(Path(image_path).read_bytes()).decode()
        response = generate(self.model_ollama,prompt,images=[image],stream=False, format=format, options=self.options, system=system)
        return response['response']
        
    def secondary_completion(self, prompt, image_path):
        instruction = (f"Generate a simple caption using the following text to guide your description. Check against the image to insure accuracy:'{prompt}'\n"
                       )
        caption = self.llm_completion(instruction, image_path, format='')
        return self.strip_text(caption)


    def caption_completion(self, prompt, image_path):
        prompt = self.strip_text(prompt)
        instruction = (f"Compare the following text with the image and remove non-visible elements: '{prompt}'\n"
                       f"Create a JSON object named 'text' containing a single string with the revised text.\n"
                       f"Be concise and use the same words and style as the text.\n"
                      )
        # instruct the LLM based on the image prompt text
        if LOGGING:
            print(f"\nInstruction:\n{instruction}\n")
        response = self.llm_completion(instruction, image_path)
        if LOGGING:
            print(f"\nElements:\n{response}\n")
        return response

    # Here is the heart of the VisionModel Magic
    def process_image(self, prompt, image_path):
        # Create list of elements
        caption = self.caption_completion(prompt, image_path)
        j = json_repair.loads(caption)
        try:
            caption = j['text']
        except:
            caption = self.caption_completion(prompt, image_path)
            j = json_repair.loads(caption)
            caption = j['text']
            
        if SECONDARY_CAPTION:
            return self.secondary_completion(caption, image_path)
        else:
            return caption


# prep prompt text for processing
# regex, format re.sub('<text to find>','<text to replace>')
def preprocess(text):
    if PREPROCESSOR:
        pattern = r'(photo of\s\w+),'
        new_text = re.sub(pattern, 'photo of', text)
        new_text = re.sub('Cybergirl', 'woman', new_text)
        new_text = re.sub('Cyberpunk man', 'man', new_text)
        new_text = re.sub('photograph', '', new_text)
        new_text = re.sub('film', '', new_text)
        new_text = re.sub('BREAK', '', new_text)
        new_text = re.sub('professional', '', new_text)
        new_text = re.sub('highly detailed', '', new_text)
        return new_text
    else:
        return text

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
                    text = f.read()
                    p = preprocess(text)
                    image_path = read_corresponding_png(filepath)
                    if image_path:
                        response = ModelClass.process_image(p, image_path)
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
