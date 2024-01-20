import tarfile
import os, os.path as osp
import json

from huggingface_hub import snapshot_download, hf_hub_download
from llava.train.simple_coyo_dataset import SimpleCoyoDataset, COYO_25M_VILA

from collections import defaultdict



def label_one_shard(gidx, dst, caption_info, out_folder="captioner"):
    webds_info = {}
    done_info = {}
    counter = defaultdict(int)
    begin_idx = dst.dataset.cum_lengths[gidx - 1] if gidx >= 1 else 0
    end_idx = dst.dataset.cum_lengths[gidx]

    for idx, inidx in enumerate(range(begin_idx, end_idx)):
        data = dst[inidx]
        url = data[".json"]["url"]
        shard = data["__shard__"]
        tar_name = osp.relpath(shard, osp.realpath(COYO_25M_VILA))
        counter[url] += 1
        
        if tar_name not in webds_info:
            webds_info[tar_name] = {}
        webds_info[tar_name][url] = caption_info[url]
        
        # if idx % 50 == 0:
        #     print(f"[{idx}-of-{len(dst)}] {tar_name} {url}")
        
    with open(osp.join(out_folder,  "_debug.json"), "w") as fp:
        json.dump(webds_info, fp)
    
    for tar_name, v in webds_info.items():
        shard_json_path = osp.join(out_folder, tar_name + ".json")
        
        # if shard in shard_info:
        #     print(f"already finished {shard_json_path}. skipping")
        os.makedirs(osp.dirname(shard_json_path), exist_ok=True)
        json.dump(
            v,
            open(shard_json_path, "w"),
            indent=2,
        )
        
        shard_info = dst.dataset.get_shard(idx)
        nsamples = shard_info[-1]["nsamples"]
        print(f"dumping {shard_json_path}, {len(v.items())} {nsamples} {begin_idx} {end_idx}")
        
        # assert len(v.items()) == nsamples, f"{len(v.items())} {nsamples} {begin_idx} {end_idx}"
    return len(v.items()), shard_info
        
if __name__ == "__main__":
    dst = SimpleCoyoDataset(data_path=COYO_25M_VILA)

    os.makedirs("captioner", exist_ok=True)

    gidx = 0
    gnshards = len(dst.dataset.shards)
    
    fpath = hf_hub_download(repo_id="Efficient-Large-Model/coyo-25m-vila-recaptioned", filename="_all.json", repo_type="dataset")
    caption_info = json.load(open(fpath, "r"))
    
    
    import concurrent.futures
    from mpi4py import MPI
    from mpi4py.futures import MPIPoolExecutor, MPICommExecutor

    # with concurrent.futures.ThreadPoolExecutor(max_workers=128) as executor:
    with MPICommExecutor() as executor:
        res = []
        jobids = list(range(gnshards))[::-1]
        for gidx in jobids:
            # label_one_shard(gidx, dst, caption_info, out_folder="captioner")
            r = executor.submit(label_one_shard, *(gidx, dst, caption_info, "captioner"))
            res.append(r)
        
        for future in concurrent.futures.as_completed(res):
            data = future.result()
            print(data)
            
    
