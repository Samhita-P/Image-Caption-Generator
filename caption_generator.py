import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Load model and processor **only once**
print("🔄 Loading model... (This happens only once)")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Set device (uses GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"✅ Model loaded successfully on {device}")

def blip_generate_caption(image_path):
    """ Generates an image caption using BLIP model. """
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(image, return_tensors="pt").to(device)

        # Faster caption generation (greedy decoding)
        out = model.generate(
            **inputs, 
            max_length=30,  # Shorter captions for speed
            do_sample=False  # Faster but deterministic output
        )

        caption = processor.decode(out[0], skip_special_tokens=True).strip()
        print(f"⚡ Debug: Raw Caption -> {caption}")

        return caption[0].upper() + caption[1:] + "."
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def adapt_caption_tone(caption, tone):
    """ Modifies caption tone (formal, casual, educational). """
    tone_mapping = {
        "formal": f"The image depicts {caption.lower()}",
        "casual": f"Whoa! Looks like {caption.lower()}",
        "educational": f"This image showcases {caption.lower()}, providing insights into the scene."
    }
    return tone_mapping.get(tone, caption)

if __name__ == "__main__":
    print("\n✨ Caption Generator Ready!")

    while True:
        image_path = input("\n📷 Enter image path (or type 'exit' to quit): ").strip()
        if image_path.lower() == "exit":
            break

        tone = input("🎭 Choose tone (formal/casual/educational): ").strip().lower()

        caption = blip_generate_caption(image_path)
        if caption:
            final_caption = adapt_caption_tone(caption, tone)
            print(f"\n✅ Generated {tone.capitalize()} Caption: {final_caption}")
