import os
import gradio as gr
import whisper
from groq import Groq
from gtts import gTTS
import tempfile

# Initialize Whisper model
model = whisper.load_model("base")

# Initialize Groq client
client = Groq(
    api_key="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",
)

# Function for speech-to-text using Whisper
def transcribe_audio(audio):
    # Transcribe the audio using Whisper
    result = model.transcribe(audio)
    return result["text"]

# Function for generating response using Llama 8B
def generate_response(transcription):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": transcription,
            }
        ],
        model="llama3-8b-8192",
    )
    return chat_completion.choices[0].message.content

# Function for text-to-speech using gTTS
def text_to_speech(response):
    tts = gTTS(response)
    temp_audio = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tts.save(temp_audio.name)
    return temp_audio.name

# Function that integrates the full pipeline
def chatbot(audio):
    transcription = transcribe_audio(audio)
    response = generate_response(transcription)
    audio_output = text_to_speech(response)
    return transcription, response, audio_output

# Gradio Interface
iface = gr.Interface(
    fn=chatbot, 
    inputs=gr.Audio(type="filepath"),  # Removed the 'source' argument
    outputs=[
        gr.Textbox(label="Transcription"), 
        gr.Textbox(label="Llama Response"), 
        gr.Audio(label="Response Audio")
    ],
    live=True
)

iface.launch()
