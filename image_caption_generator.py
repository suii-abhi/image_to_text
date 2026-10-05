import os
import tempfile
import gradio as gr
from transformers import BlipProcessor, BlipForConditionalGeneration
from gtts import gTTS
from openai import OpenAI

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY", "")
)

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def caption_image(image, style):
    if image is None:
        return "Please upload an image.", None

    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=30)
    base = processor.decode(out[0], skip_special_tokens=True)

    final = base
    if style != "Normal":
        prompts = {
            "Funny": f"Rewrite this image caption to be funny and lighthearted: '{base}'",
            "Detailed": f"Rewrite this image caption with rich detail: '{base}'",
            "Short": f"Rewrite this image caption in 3-5 words: '{base}'"
        }
        try:
            r = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[{"role": "user", "content": prompts[style]}],
                max_tokens=500,
                temperature=0.7
            )
            rewritten = (r.choices[0].message.content or "").strip()
            if rewritten:
                final = rewritten
        except Exception as e:
            print("[GROQ ERROR]", type(e).__name__, e)
            final = base

    if not final.strip():
        final = base or "An image."

    audio = None
    try:
        tts = gTTS(final, lang="en")
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(tmp.name)
        audio = tmp.name
    except Exception as e:
        print("[TTS ERROR]", type(e).__name__, e)

    return final, audio

demo = gr.Interface(
    fn=caption_image,
    inputs=[
        gr.Image(type="pil"),
        gr.Dropdown(["Normal", "Funny", "Detailed", "Short"], value="Normal", label="Style")
    ],
    outputs=[
        gr.Textbox(label="Caption"),
        gr.Audio(label="Listen", type="filepath")
    ],
    title="Image Caption Generator",
    description="Upload an image → get a caption in your chosen style + hear it read aloud."
)

if __name__ == "__main__":
    demo.launch()