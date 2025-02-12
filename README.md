# 🖼️ Context-Aware Image Caption Generator

A powerful AI-based **image captioning tool** that generates descriptions for images based on their content. This tool also adapts captions to different tones (**Formal, Casual, Educational**) and supports **batch processing** for multiple images.

## 🎯 **Features**
✅ Generate AI-powered captions for **single images**.  
✅ **Batch processing** for multiple images in a folder.  
✅ Caption tone selection: **Formal, Casual, Educational**.  
✅ **Fast & optimized** with GPU support.  
✅ **Web Interface** using **Streamlit** for easy access.  
✅ Download generated captions as a `.txt` file. 

Install Dependencies
Make sure you have Python 3.8+ installed. Then, run:
pip install -r requirements.txt

Run the Web App
streamlit run app.py
After running this command, the web app will open in your browser! 🎉

🖼️ Usage Guide
1️⃣ Single Image Captioning

Upload an image.

Choose a caption tone (Formal, Casual, Educational).

Click "Generate Caption" to get a description.

2️⃣ Batch Image Captioning
Upload multiple images at once.

Select the caption tone.

Click "Process All Images" to generate captions for all images.

Download the captions as a .txt file.

⚡ Tech Stack
Python 🐍

Hugging Face Transformers 🤗

BLIP Model (Bootstrapping Language-Image Pretraining) 🖼️

Streamlit (Web UI)

Torch & PIL (Image Processing)

🌎 Demo & Deployment
Run Locally (streamlit run app.py)
Deploy on Streamlit Cloud (Guide coming soon)

🤝 Contributing
Pull requests are welcome! If you’d like to contribute, please fork the repo and submit a PR.

📜 License
This project is open-source and available under the MIT License.

📌 Author
👩‍💻 Developed by Samhita P 🚀
