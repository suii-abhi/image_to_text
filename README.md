# Image Caption Generator

A multimodal AI app that generates image captions using BLIP, with LLM-powered style options (Funny / Detailed / Short) and text-to-speech output.

## Features
- Upload an image → get an AI-generated caption
- Choose caption style: Normal, Funny, Detailed, or Short
- Hear the caption read aloud (gTTS)

## Tech Stack
- Python
- HuggingFace Transformers (BLIP)
- Groq LLM (GPT-OSS-120B) for style rewriting
- gTTS for text-to-speech
- Gradio for the web UI

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Get a free Groq API key from console.groq.com
3. Set the key:
   - Linux/Mac: `export GROQ_API_KEY="your-key"`
   - Windows: `set GROQ_API_KEY=your-key`
4. Run: `python app.py`
