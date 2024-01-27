import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria
from transformers.models.siglip import (
    SiglipVisionModel,
    SiglipImageProcessor,
)

from transformers
from llava.model import *
from llava.model.utils import KeywordsStoppingCriteria

from PIL import Image

import os
import requests
from PIL import Image
from io import BytesIO

from tqdm import tqdm


DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def parse_annotations(anno_path, attribute_name):
    with open(anno_path) as f:
        lines = f.readlines()[1:]
    annotation_mapping = lines[0].strip().split()
    n_annotations = len(annotation_mapping)
    anno_index = annotation_mapping.index(attribute_name)
    lines = lines[1:]
    annotations = {}
    for l in lines:
        l = l.strip().split()
        assert len(l) == n_annotations + 1  # extra file name ahead
        annotations[l[0]] = l[1:][anno_index]
    return annotations


def eval_model(args):
    print(f"Evaluating {args.attribute_name}")
    annotations = parse_annotations(args.anno_path, args.attribute_name)
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if "mpt" in model_name.lower():
        model = LlavaMPTForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16, use_cache=True).cuda()
    else:
        model = LlavaLlamaForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16, use_cache=True).cuda()
    
    if "siglip" in model.config.mm_vision_tower.lower():
        image_processor = SiglipImageProcessor.from_pretrained(
            model.config.mm_vision_tower, torch_dtype=torch.float16
        )
    else:
        image_processor = CLIPImageProcessor.from_pretrained(
            model.config.mm_vision_tower, torch_dtype=torch.float16
        )

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

    vision_tower = model.get_model().vision_tower[0]
    if vision_tower.device.type == 'meta':
        if "siglip" in model.config.mm_vision_tower.lower():
            vision_tower = SiglipVisionModel.from_pretrained(vision_tower.config._name_or_path, torch_dtype=torch.float16, low_cpu_mem_usage=True).cuda()
        else:
            vision_tower = CLIPVisionModel.from_pretrained(vision_tower.config._name_or_path, torch_dtype=torch.float16, low_cpu_mem_usage=True).cuda()
        model.get_model().vision_tower[0] = vision_tower
    else:
        vision_tower.to(device='cuda', dtype=torch.float16)
    vision_config = vision_tower.config
    vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
    vision_config.use_im_start_end = mm_use_im_start_end
    if mm_use_im_start_end:
        vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
    image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2

    qs = args.query
    if mm_use_im_start_end:
        qs = qs + '\n' + DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + DEFAULT_IM_END_TOKEN
    else:
        qs = qs + '\n' + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len

    if "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt_multimodal"
    else:
        conv_mode = "multimodal"

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    inputs = tokenizer([prompt])


    image_list = sorted(os.listdir(args.image_dir))[:args.n_eval]
    print(f"Evaluating on {len(image_list)} images...")
    
    n_total = n_correct = n_unknown = 0
    for image_path in tqdm(image_list):
        gt = int(annotations[image_path])
        if args.attribute_name.startswith("No_"):
            gt = - gt
        image_path = os.path.join(args.image_dir, image_path)
        print(image_path)
    
        image = load_image(image_path)
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

        input_ids = torch.as_tensor(inputs.input_ids).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                stopping_criteria=[stopping_criteria])

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        
        if args.neg_kw in outputs.lower():
            if gt == -1:
                n_correct += 1
        elif args.pos_kw in outputs.lower():
            if gt == 1:
                n_correct += 1
        else:
            n_unknown += 1
        n_total += 1
        
        print(outputs)
        print(n_total, n_correct, n_unknown)


def eval_clip(args):
    print(f"Evaluating {args.attribute_name}")
    annotations = parse_annotations(args.anno_path, args.attribute_name)
    # Model
    disable_torch_init()
    from transformers import pipeline
    
    model_name = "openai/clip-vit-large-patch14"
    classifier = pipeline(model=model_name, task="zero-shot-image-classification")
    from PIL import Image


    # # Perform zero-shot image classification
    # inputs = processor(text=["dog", "cat"], images=image_tensor, return_tensors="pt", padding=True).to(device)
    # with torch.no_grad():
    #     logits_per_image, logits_per_text = model(**inputs)

    # # Get the predicted labels
    # predicted_labels = logits_per_text.argmax(dim=-1).tolist()[0]
    # labels = ["dog", "cat"]

    # # Print the predicted labels
    # for label_idx in predicted_labels:
    #     print("Predicted label:", labels[label_idx])

    


    image_list = sorted(os.listdir(args.image_dir))[:args.n_eval]
    print(f"Evaluating on {len(image_list)} images...")
    
    n_total = n_correct = n_unknown = 0
    for image_path in tqdm(image_list):
        gt = int(annotations[image_path])
        if args.attribute_name.startswith("No_"):
            gt = - gt
        image_path = os.path.join(args.image_dir, image_path)
        print(image_path)
    
        image = Image.open(image_path)
        
        predictions = classifier(image, candidate_labels=[args.pos_kw, args.neg_kw])[0]["label"]

        if (predictions == args.pos_kw and gt == 1) or (predictions == args.neg_kw and gt == -1):
            n_correct += 1

        n_total += 1
        
        print(n_total, n_correct, n_unknown)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str)
    parser.add_argument("--anno_path", type=str, default="/home/jil/datasets/celeba/list_attr_celeba.txt")
    parser.add_argument("--n_eval", type=int, default=500)
    parser.add_argument("--model_name", type=str, default="/home/jil/models/llava/llava-7b-v0")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--attribute_name", type=str, default="Smiling")
    parser.add_argument("--pos_kw", type=str, default="yes")
    parser.add_argument("--neg_kw", type=str, default="no")
    args = parser.parse_args()

    # eval_model(args)
    eval_clip(args)
    # python llava/eval/eval_celeba.py --image_dir /home/jil/datasets/celeba/images/ --query "Is the person smiling or not smiling?" --attribute_name Smiling --pos_kw "A photo of a person that is smiling"  --neg_kw "A photo of a person that is not smiling"