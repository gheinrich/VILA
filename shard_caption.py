import json
import os
import os.path as osp
import sys
import tarfile
from collections import defaultdict

from huggingface_hub import hf_hub_download, snapshot_download

from llava.train.simple_coyo_dataset import COYO_25M_VILA, SimpleCoyoDataset


def label_one_shard(gidx, dst, caption_info, out_folder="captioner"):
    webds_info = {}
    missing_info = []
    counter = defaultdict(int)
    begin_idx = dst.dataset.cum_lengths[gidx - 1] if gidx >= 1 else 0
    end_idx = dst.dataset.cum_lengths[gidx]

    for idx, inidx in enumerate(range(begin_idx, end_idx)):
        data = dst[inidx]
        url = data[".json"]["url"]
        shard = data["__shard__"]
        tar_name = osp.relpath(shard, osp.realpath(COYO_25M_VILA))

        shard_json_path = osp.join(out_folder, tar_name + ".json")
        # if osp.exists(shard_json_path):
        #     print(f"skipping {shard} {gidx}")
        #     return

        counter[url] += 1

        if tar_name not in webds_info:
            webds_info[tar_name] = {}
        try:
            webds_info[tar_name][url] = caption_info[url]
        except KeyError:
            # NOTE(ligeng): some urls are missing, temporailly ignore them and use original caption.
            print(data, file=sys.stderr)
            webds_info[tar_name][url] = data["text"]

        # if idx % 50 == 0:
        #     print(f"[{idx}-of-{len(dst)}] {tar_name} {url}")

    with open(osp.join(out_folder, "_debug.json"), "w") as fp:
        json.dump(webds_info, fp)

    for tar_name, v in webds_info.items():
        shard_json_path = osp.join(out_folder, tar_name + ".json")

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

    rank = 0
    world_size = 1

    if len(sys.argv) > 2:
        rank, world_size = int(sys.argv[-2]), int(sys.argv[-1])

    gidx = 0
    gnshards = len(dst.dataset.shards)
    jobids = list(range(gnshards))

    chunk = len(jobids) // world_size
    begin_idx = chunk * rank
    end_idx = chunk * (rank + 1)
    if rank == world_size - 1:
        end_idx = len(jobids)

    print(f"Ranking {rank}-of-{world_size}, {begin_idx}:{end_idx}")

    fpath = hf_hub_download(
        repo_id="Efficient-Large-Model/coyo-25m-vila-recaptioned", filename="_all.json", repo_type="dataset"
    )
    caption_info = json.load(open(fpath, "r"))

    for gidx in range(begin_idx, end_idx):
        label_one_shard(gidx, dst, caption_info, out_folder="captioner")
    exit(0)

    import concurrent.futures

    from mpi4py import MPI
    from mpi4py.futures import MPICommExecutor, MPIPoolExecutor

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
