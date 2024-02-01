import os, os.path as osp
import argparse
from hashlib import sha1, sha256

from termcolor import colored

from huggingface_hub import HfApi
from huggingface_hub.hf_api import CommitOperationAdd

max_upload_size_per_commit = 16 * 1024 * 1024 * 1024  # 16 GiB


def compute_git_hash(filename):
    with open(filename, "rb") as f:
        data = f.read()
    s = "blob %u\0" % len(data)
    h = sha1()
    h.update(s.encode("utf-8"))
    h.update(data)
    return h.hexdigest()


def compute_lfs_hash(filename):
    with open(filename, "rb") as f:
        data = f.read()
    h = sha256()
    h.update(data)
    return h.hexdigest()


if __name__ == "__main__":
    import os

    os.environ["CURL_CA_BUNDLE"] = ""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "local_folder",
        type=str,
    )
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--hash-check", action="store_true")
    parser.add_argument("--repo-type", type=str, choices=["model", "dataset"])
    parser.add_argument("--repo-org", type=str, default="Efficient-Large-Model")

    args = parser.parse_args()

    api = HfApi()

    repo_type = args.repo_type
    local_folder = args.local_folder
    if local_folder[-1] == "/":
        local_folder = local_folder[:-1]

    model_name = osp.basename(local_folder).replace("+", "-")
    repo = osp.join(args.repo_org, model_name)

    local_folder = os.path.expanduser(local_folder)
    if not api.repo_exists(repo, repo_type=repo_type):
        api.create_repo(
            repo_id=repo,
            private=True,
            repo_type=repo_type,
        )
    print(colored(f"Uploading {osp.join('https://hf.co', repo)}", "green"))

    ops = []
    commit_description = ""
    commit_size = 0
    for root, dirs, files in os.walk(local_folder, topdown=True):
        dirs.sort()
        for name in files:
            fpath = osp.join(root, name)
            rpath = osp.relpath(fpath, local_folder)
            # print(rpath)
            if "checkpoint-" in rpath:
                print(colored(f"Checkpoint detected: {rpath}, skipping", "yellow"))
                continue

            if api.file_exists(repo_id=repo, filename=rpath, repo_type=repo_type):
                # TODO: move to hash check
                if not args.hash_check:
                    print(
                        colored(
                            f"Already uploaded {rpath}, no hash check, skipping",
                            "green",
                        )
                    )
                    continue
                else:
                    hf_meta = list(
                        api.list_files_info(
                            repo_id=repo, paths=rpath, repo_type=repo_type
                        )
                    )[0]

                    if hf_meta.lfs is not None:
                        hash_type = "lfs-sha256"
                        hf_hash = hf_meta.lfs["sha256"]
                        git_hash = compute_lfs_hash(fpath)
                    else:
                        hash_type = "git-sha1"
                        hf_hash = hf_meta.blob_id
                        git_hash = compute_git_hash(fpath)

                    if hf_hash == git_hash:
                        print(
                            colored(
                                f"Already uploaded {rpath}, hash check pass, skipping",
                                "green",
                            )
                        )
                        continue
                    else:
                        print(
                            colored(
                                f"{rpath} is not same as local version, re-uploading...",
                                "red",
                            )
                        )

            operation = CommitOperationAdd(
                path_or_fileobj=fpath,
                path_in_repo=rpath,
            )
            print(colored(f"Commiting {rpath}", "green"))
            ops.append(operation)
            commit_size += operation.upload_info.size
            commit_description += f"Upload {rpath}\n"
            if len(ops) <= 8 and commit_size <= max_upload_size_per_commit:
                continue

            commit_message = "Upload files with huggingface_hub"
            commit_info = api.create_commit(
                repo_id=repo,
                repo_type=repo_type,
                operations=ops,
                commit_message=commit_message,
                commit_description=commit_description,
            )
            commit_description = ""
            ops = []
            commit_size = 0

            print(colored(f"Finish {commit_info}", "yellow"))

    # upload residuals
    commit_message = "Upload files with huggingface_hub"
    commit_info = api.create_commit(
        repo_id=repo,
        repo_type=repo_type,
        operations=ops,
        commit_message=commit_message,
        commit_description=commit_description,
    )
