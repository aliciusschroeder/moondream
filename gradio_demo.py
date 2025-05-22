import argparse
import re
from threading import Thread

import gradio as gr
import torch
from PIL import ImageDraw
from torchvision.transforms.v2 import Resize
from transformers import AutoTokenizer, TextIteratorStreamer
from moondream.torch.weights import load_weights_into_model
from moondream.torch.hf_moondream import HfConfig, HfMoondream

parser = argparse.ArgumentParser()
parser.add_argument("--cpu", action="store_true")
args = parser.parse_args()

if args.cpu:
    device = torch.device("cpu")
    dtype = torch.float32
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device != torch.device("cpu"):
        print("If you run into issues, pass the `--cpu` flag to this script.")
        print()

model_id = "vikhyatk/moondream2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
config = HfConfig()
moondream = HfMoondream(config)
# MODELPATH = "/home/alec/moondream/models/0q7mnki2.safetensors"
MODELPATH = "/home/alec/moondream/models/moondream_base.safetensors"
load_weights_into_model(MODELPATH, moondream.model)
moondream.to(device)
moondream.eval()


def answer_question(img, prompt):
    image_embeds = moondream.encode_image(img)
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
    thread = Thread(
        target=moondream.answer_question,
        kwargs={
            "image_embeds": image_embeds,
            "question": prompt,
            "tokenizer": tokenizer,
            "streamer": streamer,
        },
    )
    thread.start()

    buffer = ""
    for new_text in streamer:
        buffer += new_text
        yield buffer


def extract_floats(text):
    # Regular expression to match an array of four floating point numbers
    pattern = r"\[\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*\]"
    match = re.search(pattern, text)
    if match:
        # Extract the numbers and convert them to floats
        return [float(num) for num in match.groups()]
    return None  # Return None if no match is found


def extract_bbox(text):
    bbox = None
    if extract_floats(text) is not None:
        x1, y1, x2, y2 = extract_floats(text)
        bbox = (x1, y1, x2, y2)
    return bbox


def process_answer(img, answer):
    if extract_bbox(answer) is not None:
        x1, y1, x2, y2 = extract_bbox(answer)
        draw_image = Resize(768)(img)
        width, height = draw_image.size
        x1, x2 = int(x1 * width), int(x2 * width)
        y1, y2 = int(y1 * height), int(y2 * height)
        bbox = (x1, y1, x2, y2)
        ImageDraw.Draw(draw_image).rectangle(bbox, outline="red", width=3)
        return gr.update(visible=True, value=draw_image)

    return gr.update(visible=True, value=None)


with gr.Blocks() as demo:
    gr.Markdown(
        """
        # 🌔 moondream
        """
    )
    with gr.Row():
        prompt = gr.Textbox(label="Input Prompt", value="Describe this image.", scale=4)
        submit = gr.Button("Submit")
    with gr.Row():
        img = gr.Image(type="pil", label="Upload an Image")
        with gr.Column():
            output = gr.Markdown(label="Response")
            ann = gr.Image(visible=True, label="Annotated Image")

    submit.click(answer_question, [img, prompt], output)
    prompt.submit(answer_question, [img, prompt], output)
    output.change(process_answer, [img, output], ann, show_progress=True)

demo.queue().launch(debug=True)
