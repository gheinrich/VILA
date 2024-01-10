from random import random
import torch
from llava.train.dataset import tokenizer_image_token
from llava import conversation as conversation_lib

def call_llava_engine_df(args, sample, model, n_image_tokens, tokenizer=None, processor=None, conv_version=None):
    IMAGE_TOKEN_INDEX = 32000
    from llava.train.token_config import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

    def deal_with_prompt(input_text, mm_use_im_start_end):
        qs = input_text
        if mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        return qs

    prompt = sample['final_input_prompt']
    prompt = deal_with_prompt(prompt, model.config.mm_use_im_start_end)
    conv = conversation_lib.conv_templates[conv_version].copy()
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, n_image_tokens, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    image = sample['image']
    if image is not None:
        output_ids = model.generate(
            input_ids,
            images=image.unsqueeze(0).half().cuda(),
            do_sample=True,
            temperature=1,
            top_p=None,
            num_beams=5,
            max_new_tokens=128,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id)

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        response = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    else:  # multiple images actually
        if sample['question_type'] == 'multiple-choice':
            all_choices = sample['all_choices']
            response = random.choice(all_choices)
        else:
            response = 'INVALID GENERATION FOR MULTIPLE IMAGE INPUTS'

    return response


def llava_image_processor(raw_image, vis_processors=None):
    image_tensor = vis_processors.preprocess(raw_image, return_tensors='pt')['pixel_values'][0]
    return image_tensor