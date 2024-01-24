import base64
import io
import random

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

import argparse
from transformers import AutoTokenizer
import torch
import os
import json
from tqdm import tqdm
import pickle

from llava import LlavaLlamaForCausalLM
from llava.conversation import conv_templates
from llava.utils import disable_torch_init
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria
from transformers.models.siglip import (
    SiglipVisionModel,
    SiglipImageProcessor,
)

from torchvision import transforms
from torchvision.transforms import InterpolationMode

from llava.train.dataset import tokenizer_image_token
from llava.eval.eval_benchmarks import ToRGB, preprocess_image

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"



def decode_base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    return image

class MMBenchDataset(Dataset):
    def __init__(self,
                 data_file,
                 sys_prompt='There are several options:'):
        self.df = pd.read_csv(data_file, sep='\t')
        self.sys_prompt = sys_prompt

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        index = self.df.iloc[idx]['index']
        image = self.df.iloc[idx]['image']
        image = decode_base64_to_image(image)
        question = self.df.iloc[idx]['question']
        answer = self.df.iloc[idx]['answer'] if 'answer' in self.df.iloc[0].keys() else None
        catetory = self.df.iloc[idx]['category']
        l2_catetory = self.df.iloc[idx]['l2-category']

        option_candidate = ['A', 'B', 'C', 'D', 'E']
        options = {
            cand: self.load_from_df(idx, cand)
            for cand in option_candidate
            if self.load_from_df(idx, cand) is not None
        }
        options_prompt = f'{self.sys_prompt}\n'
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'

        hint = self.load_from_df(idx, 'hint')
        data = {
            'img': image,
            'question': question,
            'answer': answer,
            'options': options_prompt,
            'category': catetory,
            'l2-category': l2_catetory,
            'options_dict': options,
            'index': index,
            'context': hint,
        }
        
        data_sample = data
        
        image_prompt = "<image>\n"
        
        if data_sample['context'] is not None:
            qs =  image_prompt + data_sample['context'] + ' ' + data_sample['question'] + ' ' + data_sample['options']
        else:
            qs = image_prompt + data_sample['question'] + ' ' + data_sample['options']
        
        conv = conv_templates["vicuna_v1_1"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], "")
        prompt = conv.get_prompt()        
        data["prompt"] = prompt
        return data

    def load_from_df(self, idx, key):
        if key in self.df.iloc[idx] and not pd.isna(self.df.iloc[idx][key]):
            return self.df.iloc[idx][key]
        else:
            return None


def eval_model(args):
    print(f"running chunk {args.chunk_idx} of {args.num_chunks}")
    
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    model = LlavaLlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).cuda()        
    
    if "siglip" in args.model_name:
        image_processor = SiglipImageProcessor.from_pretrained(
            model.config.mm_vision_tower, torch_dtype=torch.float16
        )
    else:
        image_processor = CLIPImageProcessor.from_pretrained(
            model.config.mm_vision_tower, torch_dtype=torch.float16)

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

    vision_tower = model.get_model().vision_tower[0]
    if False:  # vision_tower.device.type == 'meta':
        if "siglip" in args.model_name:
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
    if "qwen" in model_name.lower():  # TODO: a more elegant way
        vision_config.patch_size = 28
    image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2
    
    if not hasattr(model.config, "mm_projector_type"):
        model.config.mm_projector_type = "linear"
    
    if "downsample" in model.config.mm_projector_type or "ds" in model.config.mm_projector_type:
        image_token_len = image_token_len // 4
    
    if "p32" in args.model_name:  # extra leading patches
        image_token_len += 32
    elif "se" in model.config.mm_projector_type:
        image_token_len += 2
        
    dataset = MMBenchDataset("/home/jil/datasets/mmbench/mmbench_dev_20230712.tsv")

    print(len(dataset))
    
    from llava.eval.model_vqa import get_chunk
    idx_slice = get_chunk(list(range(len(dataset))), args.num_chunks, args.chunk_idx)
    
    output_dir = os.path.join(args.model_name, "mmbench")
    os.makedirs(output_dir, exist_ok=True)
    
    # idx_slice = idx_slice[:10]  # TODO: remove me
    
    all_pred = []
    
    for i, idx in enumerate(tqdm(idx_slice)):
        data = dataset[idx]
        prompt = data["prompt"]
            
        input_ids = tokenizer_image_token(prompt, tokenizer, n_image_tokens=image_token_len, image_token_index=32000, return_tensors="pt").cuda().view(1, -1)

        image = data["img"]
        image_tensor = preprocess_image(image, image_processor)[0]
        
        # print(prompt)

        input_ids = torch.as_tensor(input_ids).cuda()

        # new stopping implementation
        class KeywordsStoppingCriteria(StoppingCriteria):
            def __init__(self, keywords, tokenizer, input_ids):
                self.keywords = keywords
                self.tokenizer = tokenizer
                self.start_len = None
                self.input_ids = input_ids

            def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
                if self.start_len is None:
                    self.start_len = self.input_ids.shape[1]
                else:
                    outputs = self.tokenizer.batch_decode(output_ids[:, self.start_len:], skip_special_tokens=True)[0]
                    for keyword in self.keywords:
                        if keyword in outputs:
                            return True
                return False

        keywords = ["</s>"]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=1024,
                use_cache=True, 
                stopping_criteria=[stopping_criteria])


        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] Sample {i}: {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0].strip()

        outputs = outputs.strip()

        pred = {
            'question': data["question"],
            'options': data["options_dict"],
            # 'answer': data["answer"],
            'prediction': outputs,
            'category': data["category"],
            'l2_category': data["l2-category"],
            'index': data["index"],
        }
        
        all_pred.append(pred)
    
    
    answers_file = os.path.join(output_dir, "mmbench-{}-shard{}.pkl".format(args.split, args.chunk_idx))
    
    with open(answers_file, "wb") as f:
        pickle.dump(all_pred, f)
    
    
def to_xlsx(args):
    # dump to excel    
    all_pred = []
    output_dir = os.path.join(args.model_name, "mmbench")
    for i_gpu in range(8):
        answers_file = os.path.join(output_dir, "mmbench-{}-shard{}.pkl".format(args.split, i_gpu))
        with open(answers_file, "rb") as f:
            all_pred.extend(pickle.load(f))
    
    answers_file = os.path.join(output_dir, "mmbench-{}.xlsx".format(args.split))
    
    all_results = []
    for data_sample in all_pred:
        result = dict()

        result['question'] = data_sample.get('question')
        result.update(data_sample['options'])
        result['prediction'] = data_sample["prediction"]
        result['category'] = data_sample['category']
        result['l2-category'] = data_sample['l2_category']
        result['index'] = data_sample['index']
        # result['split'] = data_sample['split']
        all_results.append(result)
       
    df = pd.DataFrame(all_results)
    
    with pd.ExcelWriter(answers_file, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1_1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--split", type=str, default="dev", choices=["dev", "test"])

    args = parser.parse_args()

    if args.merge:
        to_xlsx(args)
    else:
        eval_model(args)



# if __name__ == "__main__":
#     dataset = MMBenchDataset("/home/jil/datasets/mmbench/mmbench_dev_20230712.tsv")
#     print(dataset[100])