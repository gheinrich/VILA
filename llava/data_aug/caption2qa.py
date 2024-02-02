import os, os.path as osp, sys
from tqdm import tqdm
import json
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

from filelock import Timeout, FileLock
import shutil

def safely_merge_info(out_fpath, info):
    os.makedirs(osp.dirname(out_fpath), exist_ok=True)
    with FileLock(out_fpath.replace(".json", ".lock")):
        if osp.exists(out_fpath):
            new_info = json.load(
                open(out_fpath, "r+"),
            )
            info.update(new_info)
        json.dump(info, open(out_fpath + ".meta", "w+"), indent=2)
        shutil.move(out_fpath + ".meta", out_fpath)
    return info
        
def process_caption(msg):
    # msg = v['output']
    segs = []
    d = set()
    for seg in msg.split("."):
        # repetition detect
        if seg.lower() in d:
            break
        d.add(seg.lower())
        segs.append(seg)
    caption = ".".join(segs)
    return f'''Below is an image description. Please propose 3 questions and answers based on the context. Each line should start with either "question" or "answer" and there should be only single linebreak between question and answer.\n\n{caption}'''


class Cap2QADataset(Dataset):
    def __init__(self, data_path="captioner/coyo25m-0-000000.tar.json") -> None:
        caption_json = json.load(open(data_path, "r")) 
        self.captions = list(caption_json.items())
    
    def __getitem__(self, index):
        k, v = self.captions[index]
        v["cap2llm"] = process_caption(v["output"])
        return k, v
    
    def __len__(self):
        return len(self.captions)

generation_config = {
    "temperature": 0.2,
    "top_p": 0.6,
    "do_sample": True,
    "max_new_tokens": 1024,
}

def main(model_id='NousResearch/Llama-2-7b-chat-hf', data_path="captioner/coyo25m-0-000000.tar.json" ):
    dist.init_process_group()

    from llava.train.slurm_utils import get_local_rank, get_rank, get_world_size
    local_rank, rank, world_size = get_local_rank(), get_rank(), get_world_size()
    print(local_rank, rank, world_size, flush=True)

    pipe = pipeline(
        'text-generation', 
        model=model_id, 
        model_kwargs={"torch_dtype": torch.float16, "load_in_4bit": True, "device_map": f"cuda:{local_rank}"}, #"device_map": "auto"}, 
        return_full_text=False,
        repetition_penalty=1.0,
    )

    dst = Cap2QADataset(data_path=data_path)
    dloader = DataLoader(dst, batch_size=2, sampler=DistributedSampler(dst))

    output_json = {}

    save_folder = "captioner_bk" 
    save_folder = osp.join(save_folder, model_id.replace("/", "--"))
    # output_path = osp.join(save_folder, data_path.replace(".json", f"-{rank}-of-{world_size}.json"))
    output_path = osp.join(save_folder, data_path)
    os.makedirs(osp.dirname(output_path), exist_ok=True)
    
    output_json = safely_merge_info(output_path, output_json)
    
    for idx, (k, v) in enumerate(dloader):
        input_msg = v["cap2llm"]
        
        if all([url in output_json for url in k]):
            print(f"[{idx}-of-{len(dloader)}] already labeled, skip")
            continue
        
        result = pipe(input_msg, **generation_config)
        print("---" * 20, f" {idx}-of-{len(dloader)} ", flush=True)
        # print(input_msg)
        # print("***" * 40)
        # print(result)
        for url, inp, out in zip(k, input_msg, result):
            print(url, inp, out[0]["generated_text"])
            output_json[url] = {
                "caption": inp,
                "QA": out[0]["generated_text"],
            }

        if idx % 20 == 0:
            output_json = safely_merge_info(output_path, output_json)        
    
    # with open(output_path, "w") as fp:
    #     json.dump(output_json, fp, indent=2)
    output_json = safely_merge_info(output_path, output_json)


if __name__ == "__main__":
    import fire
    fire.Fire(main)
    
'''
srun --label -A llmservice_nlp_fm -N 1 \
    -p batch_block1,batch_block2,batch_block3 -t 4:00:00 \
    -J llmservice_nlp_fm:test2 --gpus-per-node 8 --exclusive \
    --pty torchrun --nproc-per-node 8  llava/data_aug/caption2qa.py --model_id=NousResearch/Llama-2-13b-chat-hf
    

JOBS_LIMIT=64  # Set your limit here
model_id=NousResearch/Llama-2-13b-chat-hf
for f in captioner/*.json; do
  while [ $(jobs -rp | wc -l) -ge $JOBS_LIMIT ]; do
    sleep 1
  done

  fname=$(echo $f | cut -d "/" -f 2)
  model=$(echo $model_id | cut -d "/" -f 2)

  # Replace this with your actual command
  echo "Processing task $f and running jobs $(jobs -rp | wc -l)"; \
  srun --label -A llmservice_nlp_fm -N 1 \
    -p batch_block1,batch_block2,batch_block3 -t 4:00:00 \
    -J llmservice_nlp_fm-dev:cap2qa-$fname-$model --gpus-per-node 8 --exclusive \
    -e slurm-logs/dev/$fname-$model-$j.err -o slurm-logs/dev/$fname-$model-$j.out \
    torchrun --nproc-per-node 8  llava/data_aug/caption2qa.py --data_path=$f --model_id=$model_id &
done
wait
'''