import os
import subprocess
import time
from argparse import ArgumentParser
from collections import deque

from llava.eval import EVAL_ROOT, TASKS


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model-path", "-m", type=str, required=True)
    parser.add_argument("--conv-mode", "-c", type=str, required=True)
    parser.add_argument("--nproc-per-node", "-n", type=int, default=8)
    parser.add_argument("--tasks", "-t", type=str)
    parser.add_argument("--include-tags", "-i", type=str)
    parser.add_argument("--exclude-tags", "-e", type=str)
    args = parser.parse_args()

    # Get the model name from the path
    model_name = os.path.basename(args.model_path).lower()

    # Filter tasks based on name and tags
    tasks = []
    for task, tags in TASKS.items():
        if args.tasks is not None and task not in args.tasks.split(","):
            continue
        tags = set(tags)
        if args.include_tags is not None and tags.isdisjoint(args.include_tags.split(",")):
            continue
        if args.exclude_tags is not None and tags.intersection(args.exclude_tags.split(",")):
            continue
        tasks.append(task)
    print(f"Running evaluation for {model_name} on {len(tasks)} tasks: {tasks}")

    # Prepare the evaluation commands
    cmds = []
    for task in tasks:
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
        # FIXME: This is a bit hacky, but it works for now
        if os.environ.get("SLURM_JOB_ID"):
            concurrency = 1
        else:
            concurrency = 10
            cmd = [f"vila-run -m eval -J {model_name}/{task}"] + cmd

        cmds.append(cmd)
    cmds = deque(cmds)

    # Prepare the environment variables
    env = os.environ.copy()
    env["NPROC_PER_NODE"] = str(args.nproc_per_node)

    # Run the commands with the specified concurrency
    processes = []
    try:
        while cmds or processes:
            while cmds and len(processes) < concurrency:
                cmd = " ".join(cmds.popleft())
                print(f"Running: {cmd}")
                processes.append(subprocess.Popen(cmd, env=env, shell=True))

            for k, process in enumerate(processes):
                if process.poll() is not None:
                    processes.pop(k)
                    break

            time.sleep(1)
    except KeyboardInterrupt:
        print("Terminating all processes...")
        for process in processes:
            process.terminate()
        for process in processes:
            process.wait()


if __name__ == "__main__":
    main()
