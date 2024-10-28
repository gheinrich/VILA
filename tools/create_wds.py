import os
import os.path as osp
import sys
import tarfile
from math import ceil, floor


def build_tar(tfiles, DIR, prefix, tar_idx=0, total_idx=None, print_freq=20):
    output_tar = osp.join(DIR, f"{osp.basename(prefix)}_{tar_idx:06d}.tar")
    tar = tarfile.open(output_tar, "w")

    for idx, f in enumerate(tfiles):
        if idx % print_freq == 0:
            print(
                output_tar,
                f"[{idx}/{len(tfiles)} | {tar_idx}-of-{total_idx}]",
                f,
            )
        # override symlinks
        try:
            tar.add(osp.realpath(f), arcname=osp.relpath(f, prefix))
        except (PermissionError, FileNotFoundError) as e:
            print(f"{f} error {e}, skip for now", file=sys.stderr)
            continue
    tar.close()
    return 0


def check_ext(file, ignored_ext=(".zip", ".tar", ".tar.gz")):
    if osp.splitext(file)[-1] in ignored_ext:
        return False
    return True


def main(
    folder="/home/ligengz/nvr_elm_llm/dataset/vila-sft",
    output_folder="/home/ligengz/nvr_elm_llm/dataset/vila-sft-tar",
    samples_per_tar=30000,
    start=0,
    seg=1000,
):
    all_files = []
    print(f"Enumerating files {folder}")
    for root, dirs, files in os.walk(folder, followlinks=True):
        for file in files:
            file_path = os.path.join(root, file)
            if check_ext(file_path):
                all_files.append(file_path)

    _all_files = set(all_files)
    print(f"Process {len(_all_files)} files")

    total_idx = ceil(len(all_files) / samples_per_tar)
    os.makedirs(output_folder, exist_ok=True)

    for shards in range(start, min(total_idx, start + seg)):
        begin_idx = shards * samples_per_tar
        end_idx = (shards + 1) * samples_per_tar
        if shards == total_idx - 1:
            end_idx = len(all_files)
        chunk_files = all_files[begin_idx:end_idx]
        print(shards, begin_idx, end_idx)
        build_tar(chunk_files, output_folder, folder, shards, total_idx)
        # input()


if __name__ == "__main__":
    import fire

    fire.Fire(main)
