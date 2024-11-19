#!/usr/bin/python3

import argparse
from collections.abc import Mapping
import copy
from datetime import datetime
import getpass
import importlib
import subprocess
import os
import platform
import socket
import sys
import shutil
from typing import List
import yaml

HOSTNAME = platform.node()
IS_AWS = HOSTNAME == 'draco-aws-login-01' or HOSTNAME.startswith('ip-')
IS_RENO = HOSTNAME == 'draco-rno-login-0001' or HOSTNAME.startswith('rno')
IS_OCI = HOSTNAME == 'cs1-login' or HOSTNAME.startswith('cs1-')
IS_BCM = 'draco-oci-login' in HOSTNAME
IS_ORD = HOSTNAME.startswith('cs-oci-ord-login')
IS_CW = HOSTNAME.startswith('cw-dfw-')
IS_PDX = HOSTNAME.startswith('cw-pdx-')

RENO_EXCLUDE_HOSTS = [
    'rno1-m02-e08-dgx1-111',
]

AWS_EXCLUDE_HOSTS = [
]

OCI_EXCLUDE_HOSTS = [
    'cs1-gpu-0012', 'cs1-gpu-0128',
]

BCM_EXCLUDE_HOSTS = [
]

ORD_EXCLUDE_HOSTS = [
]

CW_EXCLUDE_HOSTS = [
]

PDX_EXCLUDE_HOSTS = [
]


def update_name(arg_name: str, param_set: Mapping, sweep_config: Mapping):
    if sweep_config is None:
        return arg_name

    launch_config = sweep_config.get('launch_config', None)
    if launch_config is None:
        return arg_name

    # Create a copy
    param_set = dict(param_set)
    param_set['date'] = datetime.now().strftime('%m-%d-%y')

    name_map = sweep_config.get('name_map', {})

    name: str = launch_config.get('name', arg_name)

    while True:
        start_idx = name.find('{')
        if start_idx == -1:
            break
        end_idx = name.find('}', start_idx + 1)
        if end_idx == -1:
            raise ValueError(f'Unable to find matching bracket for starting at {start_idx}')

        symbol_name = name[start_idx+1 : end_idx]
        symbol_value = param_set[symbol_name]
        symbol_map = name_map.get(symbol_name, None)
        if symbol_map:
            symbol_value = symbol_map.get(symbol_value, symbol_value)

        if isinstance(symbol_value, (list, tuple)):
            symbol_value = '-'.join(str(v) for v in symbol_value)
        else:
            symbol_value = str(symbol_value)

        name = name[:start_idx] + symbol_value + name[end_idx + 1:]

    return name


def update_output_dir(output_dir, name, sweep_config):
    if sweep_config is None:
        return output_dir

    launch_config = sweep_config.get('launch_config', None)
    if launch_config is None:
        return output_dir

    append_name = launch_config.get('append_name', False)
    if append_name:
        output_dir = os.path.join(output_dir, name)

    return output_dir


def launch_job(output_dir, name, args, rest_args: List[str], param_set: Mapping, sweep_config: Mapping):
    new_name = update_name(name, param_set, sweep_config)
    if new_name != name:
        p_name = os.path.basename(output_dir)
        name = f'{p_name}_{new_name}'
    output_dir = update_output_dir(output_dir, new_name, sweep_config)

    if not args.depchain and os.path.exists(output_dir):
        print(f'Skipping "{output_dir}" because that directory already exists!')
        return

    print(f"Creating output dir '{output_dir}'...")
    os.makedirs(output_dir, exist_ok=True)
    print("Done")

    if param_set is not None:
        print(f'Sweep Config: {param_set}')
        with open(os.path.join(output_dir, 'sweep.yaml'), 'w') as fd:
            yaml.dump(param_set, fd)

    source_dir = os.path.dirname(__file__)
    if not source_dir:
        source_dir = os.getcwd()
    print('Source directory:', source_dir)

    dest_dir = os.path.join(output_dir, 'source')

    work_dir = dest_dir

    if not args.no_copy_source:
        if not os.path.exists(dest_dir):
            print('Copying source code to "{}"...'.format(dest_dir))
            shutil.copytree(source_dir, dest_dir, symlinks=True,
                ignore=shutil.ignore_patterns('logs', '.git*',
                                            '__pycache__', '.vscode', 'wandb/*')
            )
            print('Done')

            proc = subprocess.run(['bash', 'git_branch.sh'], stdout=subprocess.PIPE)

            git_info = proc.stdout.decode('utf-8').strip()

            git_dir = os.path.join(dest_dir, 'git-info')
            os.makedirs(git_dir, exist_ok=True)

            with open(os.path.join(git_dir, 'revision.txt'), 'w') as fd:
                fd.write(git_info)
                fd.write('\n')
                print('Wrote git revision info to "{}".'.format(fd.name))

            proc = subprocess.run(['git', '--no-pager', 'diff'], stdout=subprocess.PIPE)

            git_diff = proc.stdout.decode('utf-8').strip()

            with open(os.path.join(git_dir, 'delta.diff'), 'w') as fd:
                fd.write(git_diff)
                fd.write('\n')
                print('Wrote git diff to "{}".'.format(fd.name))
    else:
        work_dir = source_dir

    project_path = None

    project_path = None

    try:
        # If we're training, then inject the distributed launch code
        for train_idx, arg in enumerate(rest_args):
            if 'train.py' in arg or 'main.py' in arg:
                project_path = os.path.dirname(arg)

                OPEN_CLIP_PROJECT_NAME = 'projects/open_clip_model'
                if project_path.startswith(OPEN_CLIP_PROJECT_NAME):
                    work_dir = os.path.join(work_dir, OPEN_CLIP_PROJECT_NAME)
                    rest_args[train_idx] = arg.replace(OPEN_CLIP_PROJECT_NAME + '/', '')
                break
        else:
            raise ValueError("Not a training script!")

        dist_args = [
            'torchrun',
            '--nproc_per_node', '$SUBMIT_GPUS',
            '--master_addr', '$MASTER_ADDR',
            '--master_port', '$MASTER_PORT',
            '--nnodes', '$NUM_NODES',
            '--node_rank', '$NODE_RANK',
        ]
        rest_args = rest_args[:train_idx] + dist_args + rest_args[train_idx:]
    except:
        print('Not a training script!')

    command_args = rest_args
    if not args.no_output:
        command_args = command_args + ['--output', output_dir]

    if args.depchain:
        command_args = command_args + ['--depchain']

    command_args = subprocess.list2cmdline(command_args)
    print('command: {}'.format(command_args))

    if args.image:
        docker_image = args.image
    else:
        docker_image_dir = project_path if project_path is not None else '.'
        while True:
            docker_image_path = os.path.join(docker_image_dir, 'docker_image')
            if os.path.exists(docker_image_path):
                break
            docker_image_dir = os.path.dirname(docker_image_dir)

        with open(docker_image_path, 'r') as fd:
            docker_image = fd.read().strip().splitlines()
            docker_image = [line for line in docker_image if not line.startswith('#')]
            docker_image = ''.join(docker_image)

        if not os.path.exists(docker_image) and IS_AWS:
            docker_image += '-aws'

    print('docker image: {}'.format(docker_image))

    if not args.account:
        if os.path.exists('account'):
            with open('account', 'r') as fd:
                account = fd.read().strip()
        else:
            account = os.environ["SUBMIT_ACCOUNT"]
    else:
        account = args.account

    print('account:', account)

    num_gpus = args.gpus
    if num_gpus == 0:
        if 'batch_dgx2' in args.partition:
            num_gpus = 16
        else:
            num_gpus = 8

    duration = args.duration or 4

    ar_timer = args.autoresume_timer
    if ar_timer == 0:
        ar_timer = 30
    ar_timer = int(duration * 60 - ar_timer)

    addl_args = []
    if IS_RENO:
        exclude_list = RENO_EXCLUDE_HOSTS
    elif IS_AWS:
        exclude_list = AWS_EXCLUDE_HOSTS
    elif IS_OCI:
        exclude_list = OCI_EXCLUDE_HOSTS
    elif IS_BCM:
        exclude_list = BCM_EXCLUDE_HOSTS
        if args.interactive and num_gpus < 8:
            addl_args.extend([
                '--cpu', 20 * num_gpus,
                '--mem', 80 * num_gpus,
            ])
        else:
            addl_args.extend([
                '--exclusive',
            ])
        addl_args.append(f'--more_srun_args=--gpus-per-node={num_gpus}')
    elif IS_ORD:
        exclude_list = ORD_EXCLUDE_HOSTS
        addl_args.extend(['--exclusive', f'--more_srun_args=--gpus-per-node={num_gpus}'])
    elif IS_CW:
        exclude_list = CW_EXCLUDE_HOSTS
        addl_args.extend(['--exclusive', f'--more_srun_args=--gpus-per-node={num_gpus}'])
    elif IS_PDX:
        exclude_list = CW_EXCLUDE_HOSTS
        addl_args.extend(['--exclusive', f'--more_srun_args=--gpus-per-node={num_gpus}'])
    else:
        raise ValueError("Unknown launch node!")

    exclude_args = []
    if exclude_list:
        exclude_args.append(f'--exclude_hosts={",".join(exclude_list)}')

    if args.interactive:
        addl_args.append('--interactive')

    if IS_RENO and not args.interactive:
        # Ensure 32g machines on Reno
        addl_args.extend([
            '--constraint', 'dgx1,gpu_32gb',
        ])

    if not IS_AWS and 'MOUNTS' in os.environ:
        addl_args.extend([
            '--mounts', os.environ['MOUNTS'],
        ])

    if args.reservation:
        addl_args.extend([
            '--reservation', args.reservation,
        ])

    if name:
        addl_args.extend([
            '--name', name,
        ])
    else:
        addl_args.append('--coolname')

    login_node = args.login_node
    if login_node is not None:
        usr_name = getpass.getuser()
        addl_args.extend([
            '--submit_proxy', f'{usr_name}@{login_node}',
        ])

    dependencies = []
    if args.dependency:
        dependencies.append(args.dependency)

    if args.dependency:
        addl_args.append(f'--dependency=afterany:{args.dependency}')

    env_args = [
        # 'LD_PRELOAD=/opt/conda/lib/libgomp.so',
    ]

    WANDB_KEY = os.environ.get('WANDB_KEY', None)
    if WANDB_KEY:
        env_args.append(f'WANDB_KEY={WANDB_KEY}')

    WANDB_API_KEY = os.environ.get('WANDB_API_KEY', None)
    if WANDB_API_KEY:
        env_args.append(f'WANDB_API_KEY={WANDB_API_KEY}')

    env_args.append('NCCL_IB_QPS_PER_CONNECTION=4')
    env_args.append(f'LOGIN_NODE={socket.gethostname()}')

    if args.hf_home:
        env_args.append(f'HF_HOME={args.hf_home}')

    if env_args:
        addl_args.extend([
            '--setenv', ','.join(env_args)
        ])

    chain_length = max(1, args.depchain or 0)

    logroot = os.path.join(dest_dir, 'logs')
    os.makedirs(logroot, exist_ok=True)
    maxpart = 0
    for fd in os.listdir(logroot):
        if fd.startswith('part'):
            try:
                num = int(fd[4:])
                maxpart = max(maxpart, num + 1)
            except:
                continue

    prev_jobid = None
    for _ in range(chain_length):
        local_deps = list(dependencies)
        if prev_jobid is not None:
            local_deps.append(prev_jobid)

        local_addl_args = list(addl_args)
        if local_deps:
            depstr = f'--dependency=afterany:' + ':'.join(str(d) for d in local_deps)
            local_addl_args.append(depstr)

        submit_args = [
            'submit_job',
            '--account', account,
            '--gpu', str(num_gpus),
            '--nodes', str(args.nodes),
            '--partition', args.partition,
            '--workdir', work_dir,
            '--logdir', os.path.join(dest_dir, 'logs', f'part{maxpart}'),
            '--image', docker_image,
            '--duration', str(duration),
            '--autoresume_timer', str(ar_timer),
            *exclude_args,
            *local_addl_args,
            # '--setenv', 'CUDA_LAUNCH_BLOCKING=1',
            '-c', command_args
        ]

        print('submit_args:', submit_args)

        popen = subprocess.Popen(submit_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                universal_newlines=True)
        for stdout_line in iter(popen.stdout.readline, ""):
            print(stdout_line, end="")

            jobid_prefix = 'Submitted batch job '
            if stdout_line.startswith(jobid_prefix):
                prev_jobid = int(stdout_line[len(jobid_prefix):])
                print(f'\tFOUND JobID: {prev_jobid}')
        popen.stdout.close()
        return_code = popen.wait()

        maxpart += 1

    return return_code


def expand_recursive(val):
    expansions = [val]

    if isinstance(val, (list, tuple)):
        is_leaf = all(not isinstance(v, (list, tuple, Mapping)) for v in val)

        if is_leaf:
            expansions = val
        else:
            for i in range(len(val)):
                new_exp = []
                for exp in expansions:
                    for e in expand_recursive(exp[i]):
                        new_exp.append(exp[:i] + [e] + exp[i+1:])
                expansions = new_exp

    elif isinstance(val, Mapping):
        for k in val.keys():
            new_exp = []
            for exp in expansions:
                for e in expand_recursive(exp[k]):
                    nd = {k2: v for k2, v in exp.items()}
                    nd[k] = e
                    new_exp.append(nd)
            expansions = new_exp

    for exp in expansions:
        yield exp


def matches_filter(param_set: dict, filter: dict):
    for k, f_val in filter.items():
        if k not in param_set:
            return False
        p_val = param_set[k]
        if p_val != f_val:
            return False
    return True


def filter_param_sets(param_sets, sweep_config):
    if 'exclude' in sweep_config:
        exclude_list = sweep_config['exclude']
        new_param_sets = []
        for prms in param_sets:
            should_exclude = False
            for filter in exclude_list:
                if matches_filter(prms, filter):
                    should_exclude = True
                    break
            if not should_exclude:
                new_param_sets.append(prms)
        param_sets = new_param_sets
    return param_sets


def update_triggers(param_sets, sweep_config):
    triggers = sweep_config.get('triggers', None)
    if triggers is None:
        return param_sets

    for trigger in triggers:
        filter = trigger['filter']
        setter = trigger['setter']

        for param_set in param_sets:
            if matches_filter(param_set, filter):
                param_set.update(setter)


def get_param_sets(sweep_config):
    param_sets = []
    if 'params' in sweep_config:
        param_sets = list(expand_recursive(sweep_config['params']))

    param_sets = filter_param_sets(param_sets, sweep_config)

    update_triggers(param_sets, sweep_config)

    return param_sets


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Utility to launch jobs on the cluster.")
    parser.add_argument('-r', '--output', required=True,
                        help="Path to the output directory")
    parser.add_argument('--interactive', default=False, action='store_true',
                        help="Run the job in interactive mode")
    parser.add_argument('--partition', default='', type=str,
                        help='Which partition to launch the job on. One of: batch_16GB, batch_32GB, batch_dgx2_singlenode')
    parser.add_argument('--gpus', default=0, type=int, help='The number of GPUs to allocate')
    parser.add_argument('--nodes', default=1, type=int,
                        help='The number of nodes to allocate for the job')
    parser.add_argument('--image', type=str, default="/lustre/fsw/portfolios/llmservice/users/gheinrich/cache/vila.sqsh",
                        help='Docker image to use in lieu of default one')
    parser.add_argument('--duration', default=None, type=float,
                        help='The duration of the job in hours. Default is 8 hours.')
    parser.add_argument('--autoresume_timer', default=0, type=int,
                        help='Number of minutes to run before autoresume. If not specified, will reserve 30 minutes prior to duration ending.')
    parser.add_argument('--name', default=None, type=str,
                        help='The name to give the jobs')
    parser.add_argument('--login_node', default=None, type=str,
                        help='Launch the job via proxy to the login_node')
    parser.add_argument('--no_copy_source', default=False, action='store_true',
                        help='Don\'t copy over the source tree. Instead, use the current directory.')
    parser.add_argument('--dependency', default=None, type=str,
                        help='(Optional) A job-id that this job is dependent upon')
    parser.add_argument('--reservation', type=str, required=False,
                        help='Run the job on a specific reservation')
    parser.add_argument('--sweep', type=str, default=None,
                        help='Path to a sweep file that specifies multiple launch configurations')
    parser.add_argument('--account', type=str, default=None,
                        help='Override the default account to use')
    parser.add_argument('--no-output', default=False, action='store_true',
                        help='Don\'t forward the --output flag to the subcommand')
    parser.add_argument('--depchain', default=None, type=int,
                        help='Run this job as a dependency chain instead of using autoresume')
    parser.add_argument('--hf-home', default=None, type=str, help='Path to the HuggingFace home directory')

    # Get the output directory without fussing with the rest
    args, rest_args = parser.parse_known_args()

    partition = args.partition
    if not partition and not args.interactive:
        if IS_AWS or IS_OCI:
            partition = 'batch'
            if not args.duration:
                args.duration = 4
        elif IS_RENO:
            partition = 'batch_dgx1'
        elif IS_BCM:
            partition = 'batch_block1,batch_block3,batch_block4'
        elif IS_ORD:
            partition = 'batch'
        elif IS_CW:
            partition = 'batch'
        elif IS_PDX:
            partition = 'batch'
    args.partition = partition

    if not args.login_node:
        if HOSTNAME.startswith('rno'):
            args.login_node = 'draco-rno-login-0002'
        elif HOSTNAME.startswith('ip-'):
            args.login_node = 'draco-aws-login-01'
        elif HOSTNAME.startswith('cs1-') and HOSTNAME != 'cs1-login':
            args.login_node = 'cs1-login'

    if args.sweep:
        with open(args.sweep, 'r') as fd:
            sweep_config = yaml.load(fd, Loader=yaml.SafeLoader)

        param_sets = get_param_sets(sweep_config)
    else:
        param_sets = [None]
        sweep_config = None

    # If the output path isn't absolute, then we'll use the environment variables and project name to construct
    # the true output path
    if not os.path.isabs(args.output):
        out_root = os.environ.get('OUTPUT_DIR', None)
        if out_root is None:
            out_root = os.environ.get('SHARE_OUTPUT', None)
        if out_root is None:
            raise ValueError(f"Detected relative output path '{args.output}' but neither $OUTPUT_DIR nor $SHARE_OUTPUT exist as environment variables.")

        for arg in rest_args:
            if arg.endswith('.py'):
                break
        else:
            raise ValueError(f"Detected relative output path '{args.output}' but couldn't find the python script being executed.")

        script_dir = os.path.dirname(arg).split('/')
        module_path = '.'.join(script_dir) + '.config'
        try:
            config = importlib.import_module(module_path)

            project_name = getattr(config, 'PROJECT_NAME', None)

            if project_name is None:
                raise ValueError(f"'PROJECT_NAME' must be specified in the config file for this project.")
        except Exception:
            raise

        args.output = os.path.join(out_root, project_name, args.output)

    if not args.name:
        parts = args.output.split('/')
        last_2 = '_'.join(parts[-2:])
        args.name = last_2

    for param_set in param_sets:
        launch_job(args.output, args.name, args, rest_args, param_set=param_set, sweep_config=sweep_config)
