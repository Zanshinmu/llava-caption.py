from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
import requests
import re
import sys
import os

# We don't want a cluttered response.
def striptext(text):
    stripped_text = text.split('[/INST]')[1]
    return stripped_text
    
# prep prompt text for processing
def preprocess(text):
    new_text = re.sub('Cybergirl', 'woman', text)
    new_text = re.sub('Cyberpunk man', 'man', new_text)
    new_text = re.sub(', photograph, film, professional, highly detailed', '', new_text)
    return new_text
    
# Load the processor
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

# Load the model
model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
#model.to("mps")

# Process text with HuggingFace transformrers library
def process_image_hf(text, image):
    # Prepare the prompt
    prompt = f"[INST] <image>\nEfficiently describe image using only these words if in image:'{text}'[/INST]"
    #print (prompt)

    # Process the prompt
    inputs = processor(prompt, image, return_tensors="pt")
    #.to("mps")

    # Autoregressively complete the prompt
    processor.tokenizer.padding_side = "left"
    output = model.generate(**inputs, max_new_tokens=150)

    # Decode the output
    response = striptext(processor.decode(output[0], skip_special_tokens=True))
    #.to("mps"))
    return(response)
        
def read_corresponding_png(text_file):
    # Get the filename without extension
    filename_without_extension = os.path.splitext(text_file)[0]

    # Check if the corresponding PNG file exists
    png_filename = filename_without_extension + ".png"
    if os.path.exists(png_filename):
        # Open and display the PNG image
        img = Image.open(png_filename)
        return img
        #print(f"PNG image '{png_filename}' opened successfully.")
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
                    image = read_corresponding_png(filepath)
                    if image:
                        response = process_image_hf(text, image)
                        processed += 1
                        with open(filepath, 'w') as f:
                            f.write(response)
                            print(f"{file}: {processed} of {filecount}\n {response}\n")
                    else:
                        print(f"No coreesponding image for {file}\n")
                    
                    
if __name__ == "__main__":
    directory = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    main(directory)


