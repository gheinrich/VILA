import os
import subprocess
import time
from argparse import ArgumentParser
from collections import deque
from typing import Dict, List, Optional

from tabulate import tabulate

from llava.eval import EVAL_ROOT, TASKS
from llava.utils import io


def lstr(s: Optional[str]) -> Optional[List[str]]:
    if s is not None:
        s = s.split(",") if "," in s else [s]
    return s


def _load_results(output_dir: str, task: str) -> Optional[Dict]:
    for fname in ["results.json", "metrics.json"]:
        if os.path.exists(os.path.join(output_dir, task, fname)):
            return io.load(os.path.join(output_dir, task, fname))
    return None


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model-path", "-m", type=str, required=True)
    parser.add_argument("--conv-mode", "-c", type=str, required=True)
    parser.add_argument("--nproc-per-node", "-n", type=int, default=8)
    parser.add_argument("--tasks", "-t", type=lstr)
    parser.add_argument("--tags-include", "-ti", type=lstr)
    parser.add_argument("--tags-exclude", "-te", type=lstr)
    args = parser.parse_args()

    # Get the model name and output directory
    model_name = os.path.basename(os.path.normpath(args.model_path)).lower()
    output_dir = os.path.join("runs", "eval", model_name)

    # Filter tasks based on name and tags
    tasks = []
    for task, metainfo in TASKS.items():
        tags = set(metainfo.get("tags", []))
        if args.tasks is not None and task not in args.tasks:
            continue
        if args.tags_include is not None and tags.isdisjoint(args.tags_include):
            continue
        if args.tags_exclude is not None and tags.intersection(args.tags_exclude):
            continue
        tasks.append(task)
    print(f"Running evaluation for `{model_name}` on {len(tasks)} tasks: {tasks}")

    # Prepare the evaluation commands
    cmds = {}
    for task in tasks:
        if _load_results(output_dir, task=task):
            print(f"Skipping evaluation on `{task}` as it has already been evaluated.")
            continue

        cmd = []
        if task.startswith("lmms-"):
            cmd += [f"{EVAL_ROOT}/lmms.sh", task.replace("lmms-", ""), args.model_path]
        elif "-" in task:
            name, split = task.split("-")
            cmd += [f"{EVAL_ROOT}/{name}.sh", args.model_path, model_name, split]
        else:
            cmd += [f"{EVAL_ROOT}/{task}.sh", args.model_path, model_name]
        cmd += [args.conv_mode]

        # Wrap the command with vila-run if not running on SLURM
        if os.environ.get("SLURM_JOB_ID"):
            concurrency = 1
        else:
            concurrency = 10
            cmd = [f"vila-run -m eval -J {model_name}/{task}"] + cmd

        cmds[task] = " ".join(cmd)

    # Prepare the environment variables
    env = os.environ.copy()
    env["NPROC_PER_NODE"] = str(args.nproc_per_node)

    # Run the commands with the specified concurrency
    remaining = deque(cmds.keys())
    processes, returncodes = {}, {}
    try:
        while remaining or processes:
            while remaining and len(processes) < concurrency:
                task = remaining.popleft()
                print(f"Running evaluation on `{task}`: {cmds[task]}")
                processes[task] = subprocess.Popen(cmds[task], env=env, shell=True)

            for task, process in processes.items():
                if process.poll() is not None:
                    returncodes[task] = process.returncode
                    processes.pop(task)
                    break

            time.sleep(1)
    except KeyboardInterrupt:
        print("Terminating all processes...")
        for _, process in processes.items():
            process.terminate()
        for _, process in processes.items():
            process.wait()

    # Check the return codes
    for task, returncode in returncodes.items():
        if returncode != 0:
            print(f"Error running evaluation on `{task}`: {returncode}")

    # Collect the results and save them
    metrics = {}
    for task in tasks:
        results = _load_results(output_dir, task=task)
        if results is None:
            continue
        for name, path in TASKS[task]["metrics"].items():
            val = results
            for key in path.split("/") if "/" in path else [path]:
                val = val[key]
            metrics[f"{task}/{name}"] = val
    io.save(os.path.join(output_dir, "metrics.json"), metrics, indent=4)

    # Print the metrics in a tabular format
    print(tabulate(metrics.items(), tablefmt="simple_outline", headers=["Metric", "Value"]))


if __name__ == "__main__":
    main()
