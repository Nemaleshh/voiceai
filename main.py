import customtkinter as ctk
from PIL import Image
import sounddevice as sd
from scipy.io.wavfile import write
import os
import time
from google import genai
from google.genai import types
from txtai import LLM, RAG, Embeddings
from txtai.pipeline import Textractor

# === CONFIG ===
client=genai.Client(api_key="AIzaSyBz7xWDWXbkyA5SRFOKE0VPm4m0Uifmf0c")  # Replace with your Gemini API Key
MODEL = "gemini-1.5-flash"
AUDIO_FILENAME = "recording.wav"

# === AUDIO RECORDING ===
def record_audio(duration=5, samplerate=44100):
    recording_label.configure(text="🔴 Recording...", text_color="red")
    app.update()
    sd.default.device = None  # Let it choose the default input device
    print("Default input device:", sd.default.device)
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
    sd.wait()
    write(AUDIO_FILENAME, samplerate, recording)
    recording_label.configure(text="✅ Recording Complete", text_color="green")
    app.update()

# === GEMINI TRANSCRIBE + CLEAN + RAG ===
def process_audio():
    text_generated_box.delete("0.0", "end")
    transcript_box.delete("0.0", "end")

    with open(AUDIO_FILENAME, "rb") as f:
        audio_bytes = f.read()


    response = client.models.generate_content(
        model='gemini-1.5-flash',
        contents=[
            "Transcript the audio clip",
            types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav",)
        ]
    )
    transcript = response.text
    print(transcript)


    # Run second refinement
    final = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=transcript,
        config=types.GenerateContentConfig(
            system_instruction="Replace any sentence with a version that focuses only on Saveetha Engineering College without changing the meaning."),
    )
    refined_text = final.text
    print(refined_text)
    transcript_box.insert("0.0", refined_text)
    transcript_box.update()

    # Run through RAG
    textractor = Textractor()
    doc = textractor("Saveetha_Engineering_College.docx")
    embeddings = Embeddings(content=True)
    embeddings.index("Saveetha_Engineering_College.docx")

    llm = LLM("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    rag = RAG(embeddings, llm, output="reference")

    def prompt(question):
        return [{
            "query": question,
            "question": f"Answer the following question using only the context below. Only include information specifically discussed.\n\nquestion: {question}\ncontext:\n"
        }]

    result = rag(prompt(refined_text), maxlength=4096, pad_token_id=32000)[0]
    text_generated_box.insert("0.0", result["answer"])

# === GUI SETUP ===
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

app = ctk.CTk()
app.geometry("675x375")
app.title("Voice AI")
app.configure(fg_color="black")
app.resizable(False, False)

title_label = ctk.CTkLabel(app, text="Voice AI", font=("Bruno Ace SC", 28), text_color="#A259FF")
title_label.place(x=20, y=10)

image_path = "elements/line1.png"
png_image1 = ctk.CTkImage(dark_image=Image.open(image_path), size=(675, 1))
image_label1 = ctk.CTkLabel(app, image=png_image1, text="", bg_color="black")
image_label1.place(x=0, y=40)

recording_label = ctk.CTkLabel(app, text="Idle", font=("Jura", 15), text_color="#CCCCCC")
recording_label.place(x=20, y=60)

# Buttons
start_button = ctk.CTkButton(app, font=("Jura", 14), text="Start", width=100,
                             fg_color="transparent", border_color="white",
                             border_width=1, corner_radius=20, hover_color="white",
                             command=lambda: [record_audio(), process_audio()])
start_button.place(x=20, y=95)

stop_button = ctk.CTkButton(app, font=("Jura", 14), text="Stop", width=100,
                            fg_color="transparent", border_color="white",
                            border_width=1, corner_radius=20, hover_color="white")
stop_button.place(x=140, y=95)

# Text boxes
text_generated_label = ctk.CTkLabel(app, text="Text Generated", font=("Jura", 15))
text_generated_label.place(x=375, y=60)

text_generated_box = ctk.CTkTextbox(app, font=("Jura", 12), width=270, height=170,
                                    fg_color="transparent", border_color="white",
                                    border_width=1, corner_radius=0)
text_generated_box.place(x=375, y=95)

transcript_label = ctk.CTkLabel(app, text="Tran-scripted Text", font=("Jura", 14))
transcript_label.place(x=20, y=135)

transcript_box = ctk.CTkTextbox(app, font=("Jura", 11), width=300, height=100,
                                fg_color="transparent", border_color="white",
                                border_width=1, corner_radius=0)
transcript_box.place(x=20, y=165)

speech_generated_label = ctk.CTkLabel(app, text="Speech Generated", font=("Jura", 14))
speech_generated_label.place(x=20, y=300)

app.mainloop()
