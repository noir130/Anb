import gradio as gr
from transformers import pipeline
import os

# Load model with CPU optimization
model = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device_map="cpu",
    torch_dtype="auto"
)

def chat(message, history):
    response = model(message, max_new_tokens=80)[0]['generated_text']
    return response

gr.ChatInterface(chat).launch(
    server_name="0.0.0.0",
    server_port=int(os.getenv("PORT", 7860))
