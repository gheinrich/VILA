import numpy as np
import gradio as gr

import os, os.path as osp
import json

import random

model1 = "checkpoints/vila-7B"
model2 = "checkpoints/vicuna-13b-clip336-mmc4sub+coyo-finetune-llava15+vflan+sharegpt4v-nosqa-linear-captioner1"

dataset = "sam"
default_img = "/lustre/fs8/portfolios/llmservice/users/jasonlu/vlm_datasets/ShareGPT4V/data/sam/images/sa_394568.jpg"

info1 = json.load(
    open(osp.join("captioner", f"{dataset}-{osp.basename(model1)}.json"), "r")
)

info2 = json.load(
    open(osp.join("captioner", f"{dataset}-{osp.basename(model2)}.json"), "r")
)

overlap_keys = set(info1.keys()) & set(info2.keys())

new_keys = set()
for full_fpath in overlap_keys:
    new_keys.add(
        # full_fpath.replace('/lustre/fs8/portfolios/llmservice/users/jasonlu/vlm_datasets/', '')
        full_fpath,
    )


def get_image_list(x):
    l = overlap_keys()
    return l[0]


def compare_list(dataset, image_id):
    if image_id is None or len(image_id) <= 5:
        image_id = random.choice(list(new_keys))

    if image_id not in new_keys:
        return [
            None,
            "Failed",
            "Invalid file",
            "Invalid file",
        ]

    return [
        image_id,
        image_id,
        info1[image_id]["output"],
        info2[image_id]["output"],
    ]

    return [
        "https://camo.githubusercontent.com/dfc47ba678ef9055a771b0857560f28494f03b13bb59aa5e3b1bf1c9fa74bc22/68747470733a2f2f70696373756d2e70686f746f732f3436302f333030",
        "another testing",
        "hello",
        "world",
    ]


with gr.Blocks(
    title="VILA Captioner Comparison",
    theme=gr.themes.Monochrome(),
    css="footer{display:none !important}",
) as demo:
    gr.Markdown("Quick caption visualization for VILA.")

    with gr.Row():
        with gr.Column():
            dataset = gr.Radio(
                ["sam"], label="The dataset for visualization", value="sam"
            )
        with gr.Column():
            image_id = gr.Text(
                label="Input an image fpath for comparison. Will randomly pick one if set to None.",
            )
    compare_btn = gr.Button("Compare")

    with gr.Row():
        output_image = gr.Image(
            default_img,
            label="Image Visualization",
            width="30vw",
        )
        with gr.Column():
            caption_output = gr.TextArea(
                value=info1[default_img]["output"], label="vila output"
            )
            vila_output = gr.TextArea(
                value=info2[default_img]["output"], label="captioner output"
            )
            output_text = gr.Text(value=default_img, label="image id")

    compare_btn.click(
        compare_list,
        [dataset, image_id],
        [output_image, output_text, caption_output, vila_output],
    )

demo.launch(share=True)
