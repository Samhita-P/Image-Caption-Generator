import streamlit as st
import torch
import os
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load the BLIP model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# Function to generate a caption
def blip_generate_caption(image):
    inputs = processor(image, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_length=30, do_sample=False)
    caption = processor.decode(output[0], skip_special_tokens=True).strip()
    return caption[0].upper() + caption[1:] + "."

# Function to adapt the caption's tone
def adapt_caption_tone(caption, tone):
    tone_mapping = {
        "formal": f"The image depicts {caption.lower()}",
        "casual": f"Whoa! Looks like {caption.lower()}",
        "educational": f"This image showcases {caption.lower()}, providing insights into the scene."
    }
    return tone_mapping.get(tone, caption)

# Streamlit UI
st.title("🖼️ AI Image Caption Generator")
st.write("Upload an image (or folder) to generate captions with different tones.")

# Mode selection (Single Image or Batch)
mode = st.radio("Select Mode:", ["Single Image", "Batch Processing"])

# --- Single Image Mode ---
if mode == "Single Image":
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    tone = st.selectbox("Choose Caption Tone:", ["formal", "casual", "educational"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Generate Caption"):
            with st.spinner("Generating Caption..."):
                raw_caption = blip_generate_caption(image)
                final_caption = adapt_caption_tone(raw_caption, tone)
                st.success(f"**Generated {tone.capitalize()} Caption:** {final_caption}")

# --- Batch Processing Mode ---
elif mode == "Batch Processing":
    uploaded_files = st.file_uploader("Upload Multiple Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    tone = st.selectbox("Choose Caption Tone:", ["formal", "casual", "educational"])
    
    if uploaded_files and st.button("Process All Images"):
        captions = {}
        with st.spinner("Processing Images..."):
            for uploaded_file in uploaded_files:
                image = Image.open(uploaded_file).convert("RGB")
                raw_caption = blip_generate_caption(image)
                final_caption = adapt_caption_tone(raw_caption, tone)
                captions[uploaded_file.name] = final_caption

        # Display results
        st.success("Batch Processing Completed! Here are the results:")
        for img_name, caption in captions.items():
            st.write(f"📸 **{img_name}**: {caption}")

        # Save captions to file
        with open("captions_output.txt", "w") as f:
            for img, caption in captions.items():
                f.write(f"{img}: {caption}\n")

        # Provide Download Link
        with open("captions_output.txt", "rb") as f:
            st.download_button("📥 Download Captions", f, file_name="captions_output.txt")

