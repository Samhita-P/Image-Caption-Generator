import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os

# Load the BLIP model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device set to use: {device}")

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# Function to generate caption
def blip_generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt").to(device)
    output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

# Function to adapt tone
def adapt_caption_tone(caption, tone):
    if tone == "formal":
        return f"The image depicts {caption}."
    elif tone == "casual":
        return f"Looks like {caption}!"
    elif tone == "educational":
        return f"This image represents {caption}, a concept often seen in real-life scenarios."
    else:
        return caption  # Default

# Function to process all images in a folder
def process_images_in_folder(folder_path, tone):
    captions = {}
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Process only images
            image_path = os.path.join(folder_path, filename)
            print(f"Processing: {filename}...")

            raw_caption = blip_generate_caption(image_path)
            final_caption = adapt_caption_tone(raw_caption, tone)
            captions[filename] = final_caption

            print(f"Generated Caption: {final_caption}\n")

    # Save results to a text file
    with open("captions_output.txt", "w") as f:
        for img, caption in captions.items():
            f.write(f"{img}: {caption}\n")

    print("All captions saved to captions_output.txt ✅")

# User input for folder path and tone
folder_path = input("Enter folder path containing images: ").strip()
tone = input("Choose tone (formal/casual/educational): ").strip().lower()

# Run batch processing
process_images_in_folder(folder_path, tone)
