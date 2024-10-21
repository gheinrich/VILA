import json
import os
import os.path as osp
import random

import gradio as gr

info = {
    "chartqa_train_18k/15952-0/0": {
        "uid": "chartqa_train_18k/15952-0",
        "path": "/home/jasonlu/workspace/InternVL/internvl_chat/playground/data/chartqa/train/png/multi_col_80596.png",
        "full_text": "What was the percentage of people aged 15 to 29 who accessed mobile internet in the three months prior to this survey? Answer the question using a single word or phrase.\n######\n95.1",
        "raw_question": "What was the percentage of people aged 15 to 29 who accessed mobile internet in the three months prior to this survey? Answer the question using a single word or phrase.",
        "raw_answer": "95.1",
        "recaptioned_prompt": "What was the percentage of people aged 15 to 29 who accessed mobile internet in the three months prior to this survey? While the answer should be (95.1) now let's think step by step to explain the answer and each step shall be separated by '#####' at the beginning.In the last line, answer the question using a single word or phrase.",
        "recaptioned_answer": "##### First, identify the relevant year for the data, which is 2019.\n\n##### Next, locate the blue bar on the graph corresponding to the 15-29 age group for 2019.\n\n##### Observe the height of the blue bar to determine the percentage it represents.\n\n##### According to the graph, the height of the blue bar for 2019 reaches approximately 95%.\n\n##### This percentage represents the proportion of people aged 15 to 29 who accessed mobile internet within the last three months of the survey.\n\n##### Therefore, confirming this matches the expected figure of 95.1%.\n\nAnswer: 95.1%",
        "labeler": "gpt-4o",
    },
    "chartqa_train_18k/1797-0/0": {
        "uid": "chartqa_train_18k/1797-0",
        "path": "/home/jasonlu/workspace/InternVL/internvl_chat/playground/data/chartqa/train/png/multi_col_21040.png",
        "full_text": "What year did the survey ask respondents about their plans for trade shows? Answer the question using a single word or phrase.\n######\n2021\n######\nWhat year did the survey ask respondents about their plans for trade shows?\n######\nThe survey asked respondents about their plans for trade shows in 2021.\n######\nHow many regional shows did industry professionals attend in 2020? Answer the question using a single word or phrase.\n######\n7.8\n######\nHow many regional shows did industry professionals attend in 2020?\n######\nIndustry professionals attended 7.8 regional shows in 2020.\n######\nWhich year has the least number of international trade shows? Answer the question using a single word or phrase.\n######\n2020\n######\nWhich year has the least number of international trade shows?\n######\nIn 2020, there were the least number of international trade shows.\n######\nWhat is the total number of shows in the year 2017? Answer the question using a single word or phrase.\n######\n39.1\n######\nWhat is the total number of shows in the year 2017?\n######\nThere were a total of 39.1 shows in the year 2017.",
        "raw_question": "What is the total number of shows in the year 2017?",
        "raw_answer": "There were a total of 39.1 shows in the year 2017.",
        "recaptioned_prompt": "What is the total number of shows in the year 2017? While the answer should be (There were a total of 39.1 shows in the year 2017.) now let's think step by step to explain the answer and each step shall be separated by '#####' at the beginning.In the last line, answer the question using a single word or phrase.",
        "recaptioned_answer": "##### First, identify the number of Regional shows in 2017.\nThe number of Regional shows in 2017 is 23.5.\n\n##### Next, identify the number of National shows in 2017.\nThe number of National shows in 2017 is 12.1.\n\n##### Then, identify the number of International shows in 2017.\nThe number of International shows in 2017 is 3.5.\n\n##### Now, sum the numbers of Regional, National, and International shows for 2017.\n23.5 (Regional) + 12.1 (National) + 3.5 (International) = 39.1\n\n##### Therefore, the total number of shows in 2017 was 39.1.\n\n39.1",
        "labeler": "gpt-4o",
    },
    "chartqa_train_18k/1797-123": {
        "uid": "chartqa_train_18k/1797-0",
        "path": "/home/jasonlu/workspace/InternVL/internvl_chat/playground/data/chartqa/train/png/multi_col_21040.png",
        "full_text": "What year did the survey ask respondents about their plans for trade shows? Answer the question using a single word or phrase.\n######\n2021\n######\nWhat year did the survey ask respondents about their plans for trade shows?\n######\nThe survey asked respondents about their plans for trade shows in 2021.\n######\nHow many regional shows did industry professionals attend in 2020? Answer the question using a single word or phrase.\n######\n7.8\n######\nHow many regional shows did industry professionals attend in 2020?\n######\nIndustry professionals attended 7.8 regional shows in 2020.\n######\nWhich year has the least number of international trade shows? Answer the question using a single word or phrase.\n######\n2020\n######\nWhich year has the least number of international trade shows?\n######\nIn 2020, there were the least number of international trade shows.\n######\nWhat is the total number of shows in the year 2017? Answer the question using a single word or phrase.\n######\n39.1\n######\nWhat is the total number of shows in the year 2017?\n######\nThere were a total of 39.1 shows in the year 2017.",
        "raw_question": "What is the total number of shows in the year 2017?",
        "raw_answer": "oh yeyeyeye.",
        "recaptioned_prompt": "What is the total number of shows in the year 2017? While the answer should be (There were a total of 39.1 shows in the year 2017.) now let's think step by step to explain the answer and each step shall be separated by '#####' at the beginning.In the last line, answer the question using a single word or phrase.",
        "recaptioned_answer": "##### First, identify the number of Regional shows in 2017.\nThe number of Regional shows in 2017 is 23.5.\n\n##### Next, identify the number of National shows in 2017.\nThe number of National shows in 2017 is 12.1.\n\n##### Then, identify the number of International shows in 2017.\nThe number of International shows in 2017 is 3.5.\n\n##### Now, sum the numbers of Regional, National, and International shows for 2017.\n23.5 (Regional) + 12.1 (National) + 3.5 (International) = 39.1\n\n##### Therefore, the total number of shows in 2017 was 39.1.\n\n39.1",
        "labeler": "gpt-4o",
    },
}


info = json.load(open("recaptioned.json"))
info = list(info.values())


def update_examples(
    index,
):
    return (
        # "https://www.gradio.app/_app/immutable/assets/header-image.DJQS6l6U.jpg",
        info[index]["path"],
        info[index]["uid"] + "\n#####\n" + info[index]["raw_question"] + "\n#####\n" + info[index]["raw_answer"],
        info[index]["recaptioned_prompt"],
        info[index]["recaptioned_answer"],
    )


with gr.Blocks() as demo:
    with gr.Row():
        rand = gr.Button(value="Random")
        mark = gr.Button(value="Mark")
    slider = gr.Slider(maximum=len(info) - 1, step=1)

    with gr.Row():
        with gr.Column():
            image = gr.Image(value="https://www.gradio.app/_app/immutable/assets/header-image.DJQS6l6U.jpg", height=300)
            descriptions = gr.Textbox(label="Original Question", placeholder="", interactive=False)
        with gr.Column():
            recaptioned_prompt = gr.Textbox(label="recaptioned_prompt", placeholder="", interactive=False)
            recaptioned_answer = gr.Textbox(label="recaptioned_answer (gpt-4o)", placeholder="", interactive=False)

    slider.change(
        update_examples,
        inputs=[slider],
        outputs=[image, descriptions, recaptioned_prompt, recaptioned_answer],
    )

    def rand_slider_index():
        idx = random.randint(0, len(info) - 1)
        gr.Info(f"randomly viz {idx} !", duration=3)
        return idx

    rand.click(rand_slider_index, outputs=slider)

    def mark_special(index):
        os.makedirs("tmp", exist_ok=True)
        with open(f"tmp/{index}.txt", "w") as f:
            f.write("")
        gr.Info(f"Marked {index} !", duration=3)

    mark.click(mark_special, inputs=slider)
demo.launch(share=True)
