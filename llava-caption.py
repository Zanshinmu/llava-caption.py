from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, LlavaNextConfig
from PIL import Image
import torch
import requests
import re
import sys
import os

# Set up hf transformers
MODEL_PATH = "llava-hf/llava-v1.6-mistral-7b-hf"
# Prompt Preprocessor defaults to off
PREPROCESSOR = False

class HFModel:
    def __init__(self):
        # Define your machine learning objects in the constructor
        # set torch device from env, default is cpu
        self.device = torch.device(os.environ.get('TORCH_DEVICE', "cpu"))
        if self.device.type == "mps":
            self.dtype = torch.float16
        elif self.device.type == "cpu":
            self.dtype = torch.float32
        elif self.device.type == "cuda":
            self.dtype = torch.bfloat16
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            MODEL_PATH, torch_dtype=self.dtype, low_cpu_mem_usage=True, device_map=self.device,
        )
        self.processor = LlavaNextProcessor.from_pretrained(MODEL_PATH)
        self.processor.tokenizer.padding_side = "left"
        print(f"Model initialized for device '{self.device}'\n")

# We don't want a cluttered response.
def striptext(text):
    try:
        stripped_text = text.split('[/INST]')[1]
    except:
        return text
    return stripped_text


# prep prompt text for processing
def preprocess(text):
    new_text = re.sub('Cybergirl', 'woman', text)
    new_text = re.sub('Cyberpunk man', 'man', new_text)
    new_text = re.sub(', photograph, film, professional, highly detailed', '', new_text)
    return new_text


def resize_image(image):
    # image must be resized for model
    # Within model limits and preserves aspect ratio
    # note we are using thumbnail here because it instantiates a new resized image inline without modifying the original
    max_size = (670, 670)

    image.thumbnail(max_size, Image.LANCZOS)
    return image


# Process text with HuggingFace transformers library and hopefully MPS/CUDA
def process_image_hf(text, image):
    # Prepare the prompt
    prompt = f"[INST] <{image}>\nEfficiently describe image using only these words if in image:'{text}'[/INST]"

    # process the prompt

    inputs = hfmodel.processor(prompt, image, return_tensors="pt", padding=True).to(hfmodel.device)

    # Autoregressively complete the prompt
    hfmodel.model.config.image_token_index = 1
    output = hfmodel.model.generate(**inputs, max_new_tokens=250, pad_token_id=hfmodel.processor.tokenizer.pad_token_id)

    # Decode the output
    response = striptext(hfmodel.processor.decode(output[0], skip_special_tokens=True))

    return response


def read_corresponding_png(text_file):
    # Get the filename without extension
    filename_without_extension = os.path.splitext(text_file)[0]

    # Check if the corresponding PNG file exists
    png_filename = filename_without_extension + ".png"
    if os.path.exists(png_filename):
        # Open and display the PNG image
        original = Image.open(png_filename)
        img = resize_image(original)
        img.save("temp.png")
        llavaimage = Image.open("temp.png")
        original.close()

        return llavaimage
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
                    if (PREPROCESSOR):
                        text = preprocess(f.read())
                    else:
                        text = f.read()
                    image = read_corresponding_png(filepath)
                    if image:
                        response = process_image_hf(text, image)
                        image.close()
                        processed += 1
                        with open(filepath, 'w') as f:
                            f.write(response)
                            print(f"{file}: {processed} of {filecount}\n {response}\n")
                    else:
                        print(f"No corresponding image for {file}\n")


if __name__ == "__main__":
    hfmodel = HFModel()
    directory = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    main(directory)
