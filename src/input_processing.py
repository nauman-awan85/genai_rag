import torch
import whisper
import numpy as np
import soundfile as sf
import librosa
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


# Whisper Initialization (Audio to Text)
whisper_device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisper.load_model("base", device=whisper_device)
print("Whisper model loaded successfully.")


# BLIP Initialization (Image Captioning)
device = "cuda" if torch.cuda.is_available() else "cpu"
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)
blip_model.eval()
print("BLIP-base model loaded successfully.")


# Audio to Text using Whisper, Convert audio into text using Whisper.
def speech_to_text(audio_file):

    if hasattr(audio_file, "read"):
        import io
        data = io.BytesIO(audio_file.read())
        audio, sr = sf.read(data, dtype="float32")
    else:
        audio, sr = sf.read(audio_file, dtype="float32")

    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

# Transcribe by Whisper
    result = whisper_model.transcribe(audio, fp16=False)
    text = result["text"].strip()
    print(f"Transcription: {text}")

    return text, True


# Image to Text using BLIP, Generate caption for an image using BLIP
def image_to_text(image_path):

    image = Image.open(image_path).convert("RGB")
    inputs = blip_processor(image, return_tensors="pt").to(device)

    with torch.no_grad():
        out = blip_model.generate(**inputs, max_new_tokens=60)

    caption = blip_processor.decode(out[0], skip_special_tokens=True).strip()

    if any(word in caption.lower() for word in ["diagram", "graph", "code", "table", "chart"]):
        caption += " — likely a technical or document image."
    else:
        caption += " — likely a real-world photo."

    return caption


# Text Input Handling
def handle_text_input(text):
    return text.strip()



# https://huggingface.co/docs/transformers/main/en/model_doc/blip

# https://huggingface.co/docs/transformers/main/en/model_doc/whisper