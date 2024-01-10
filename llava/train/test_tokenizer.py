from transformers import AutoTokenizer
import torch

def tokenizer_image_token(prompt, tokenizer, n_image_tokens=256, image_token_index=32000, return_tensors=None):
    # TODO: feed in image token instead of 32000 hardcoded
    
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + n_image_tokens)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids



tokenizer = AutoTokenizer.from_pretrained("/home/jil/models/llama-2-hf/llama-2-7b",
                                          use_fast=False, legacy=False)



text = "Hi <image> describe the <image>\n<image>\n<image>\n image in a single sentence, making sure to mention the text present in the image."


print(tokenizer_image_token(text, tokenizer, n_image_tokens=4))

