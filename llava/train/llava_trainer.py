import os
import torch
import torch.nn as nn

from transformers import Trainer, PreTrainedModel
from typing import Dict, Optional, Sequence
from torch.utils.data import DistributedSampler, RandomSampler
from typing import Iterator, Sized


def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Args:
        model (`torch.nn.Module`): The model to unwrap.
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model


class LocalDistributedSampler(DistributedSampler):
    def __iter__(self):
        import random
        import math

        indices = list(range(len(self.dataset)))

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size /
                            len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # let's first do subsample
        indices = indices[self.rank *
                          self.num_samples:(self.rank + 1) * self.num_samples]

        if self.shuffle:
            random.seed(self.seed + self.epoch)
            random.shuffle(indices)

        return iter(indices)



class LocalDistributedSamplerAugCoyo(DistributedSampler):
    def __init__(self, dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False,
                 batch_size=None,
                 # NOTE: this is the total size but not per-worker
                 sample_len_list = None,
                #  mmc4_samples=None, coyo_samples=None, laion_samples=None,
                 # means that we are sampling 1 mmc4 + 1 coyo, and accumulate gradients
                 force_accumulation=True,
                 ) -> None:
        import math
        import torch.distributed as dist

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = True  # always True
        
        # NOTE: org_ is without drop last
        self.org_sample_len_list = self.per_replica_samples = sample_len_list

        assert sum(sample_len_list) == len(self.dataset), f"{sum(sample_len_list)} != {len(self.dataset)}"

        self.batch_size = batch_size
        self.global_batch_size = batch_size * num_replicas

        if self.drop_last:  # type: ignore[arg-type]
            self.per_replica_samples = [
                sample_len // (self.num_replicas * batch_size) * batch_size
                for sample_len in self.per_replica_samples
            ]
            self.num_samples = sum(self.per_replica_samples)
        else:
            raise NotImplementedError
    
        self.total_size = self.num_samples * self.num_replicas
        self.total_samples = [samples * self.num_replicas for samples in self.per_replica_samples]
        
        self.shuffle = shuffle
        self.seed = seed

        # whether to force accumulate
        self.force_accumulation = force_accumulation

    def __iter__(self):
        import random
        indices = list(range(len(self.dataset)))

        indices_list = []
        for i in range(len(self.org_sample_len_list)):
            indices_list.append(indices[
                sum(self.org_sample_len_list[:i]): sum(self.org_sample_len_list[:i]) + self.total_samples[i]
                ])
            
        # 1. split the full indices first (note: without drop last at this moment)
        indices_list = []
        for i in range(len(self.org_sample_len_list)):
            indices_list.append(indices[
                sum(self.org_sample_len_list[:i]): sum(self.org_sample_len_list[:i]) + self.total_samples[i]
                ])

        assert sum([len(indices) for indices in indices_list]) == self.total_size, \
            (sum([len(indices) for indices in indices_list]), self.total_size)

        # let's first do subsample
        for idx, indices in enumerate(indices_list):
            indices_list[idx] = indices[self.rank *
                            self.per_replica_samples[idx]:(self.rank + 1) * self.per_replica_samples[idx]]

        random.seed(self.seed + self.epoch)
        for indice in range(len(indices_list)):
            random.shuffle(indices_list[indice])

        indices_list = sorted(indices_list, key=lambda x:-len(x))
        all_indices = [-1] * self.num_samples
        indices_available = list(range(self.num_samples))
        for indice in indices_list:
            original_indices = range(len(indice))
            transformed_indices = [idx * len(indices_available) // len(indice) for idx in original_indices]
            mapped_indices = [indices_available[idx] for idx in transformed_indices]
            # update indices_available
            for idx in reversed(transformed_indices):
                del indices_available[idx]
            for i, idx in enumerate(mapped_indices):
                all_indices[idx] = indice[i]
        assert -1 not in all_indices
        
        return iter(all_indices)


class LLaVATrainer(Trainer):

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            # Save the model
            _state_dict = state_dict
            if _state_dict is None:
                # Only save the model itself if we are using distributed training
                model_to_save = unwrap_model(self.model)
                _state_dict = model_to_save.state_dict()

            weight_to_save = {}
            keys_to_match = ['mm_projector', 'embed_tokens', 'embed_in']
            for k, v in _state_dict.items():
                if any(key_match in k for key_match in keys_to_match):
                    weight_to_save[k] = v

            current_folder = output_dir.split('/')[-1]
            parent_folder = os.path.dirname(output_dir)
            if 'checkpoint-' in current_folder:
                mm_projector_folder = os.path.join(
                    parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(
                    mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(
                    output_dir, f'mm_projector.bin'))

        from peft.peft_model import PeftModelForCausalLM
        if isinstance(self.model, PeftModelForCausalLM):
            self.model.save_pretrained(output_dir, state_dict=state_dict)
        elif not isinstance(self.model, PreTrainedModel) and isinstance(unwrap_model(self.model), PeftModelForCausalLM):
            unwrap_model(self.model).save_pretrained(
                output_dir, state_dict=state_dict)
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)

    def create_optimizer_fclr10(self):

        # customize the function to support projector learning rate x 10
        from transformers.trainer import is_sagemaker_mp_enabled, get_parameter_names, ALL_LAYERNORM_LAYERS, ShardedDDPOption
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        
        LR10_KEYWORDS = ["mm_projector", "_visual."]
        def _should_lr10(n):
            return any([k in n for k in LR10_KEYWORDS])

        if self.optimizer is None:
            decay_parameters = get_parameter_names(
                opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [
                name for name in decay_parameters if "bias" not in name]

            print("* Using 10x learning rate for projector...")

            # just for sanity check
            decay_params = [p for n, p in opt_model.named_parameters() if (
                n in decay_parameters and p.requires_grad)]
            no_decay_params = [p for n, p in opt_model.named_parameters() if (
                n not in decay_parameters and p.requires_grad)]
            org_n_params = len(decay_params) + len(no_decay_params)
            print("nparams before", len(decay_params), len(no_decay_params))

            decay_params = [p for n, p in opt_model.named_parameters() if (
                n in decay_parameters and p.requires_grad and not _should_lr10(n))]
            no_decay_params = [p for n, p in opt_model.named_parameters() if (
                n not in decay_parameters and p.requires_grad and not _should_lr10(n))]

            projector_decay_params = [p for n, p in opt_model.named_parameters() if (
                n in decay_parameters and p.requires_grad and _should_lr10(n))]
            projector_no_decay_params = [p for n, p in opt_model.named_parameters() if (
                n not in decay_parameters and p.requires_grad and _should_lr10(n))]

            print("nparams after", len(decay_params), len(no_decay_params), len(
                projector_decay_params), len(projector_no_decay_params))
            assert len(decay_params) + len(no_decay_params) + \
                len(projector_decay_params) + \
                len(projector_no_decay_params) == org_n_params

            print("decay_params", [n for n, p in opt_model.named_parameters() if (
                n in decay_parameters and p.requires_grad and not _should_lr10(n))])
            print("no_decay_params", [n for n, p in opt_model.named_parameters() if (
                n not in decay_parameters and p.requires_grad and not _should_lr10(n))])
            print("projector_decay_params", [n for n, p in opt_model.named_parameters() if (
                n in decay_parameters and p.requires_grad and _should_lr10(n))])
            print("projector_no_decay_params", [n for n, p in opt_model.named_parameters(
            ) if (n not in decay_parameters and p.requires_grad and _should_lr10(n))])

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                self.args)

            lr = optimizer_kwargs.pop('lr')

            optimizer_grouped_parameters = [
                {
                    "params": decay_params,
                    "weight_decay": self.args.weight_decay,
                    "lr": lr,
                },
                {
                    "params": no_decay_params,
                    "weight_decay": 0.0,
                    "lr": lr,
                },
                {
                    "params": projector_decay_params,
                    "weight_decay": self.args.weight_decay,
                    "lr": lr * 10,
                },
                {
                    "params": projector_no_decay_params,
                    "weight_decay": 0.0,
                    "lr": lr * 10,
                },
            ]

            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                raise NotImplementedError
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(
                    optimizer_grouped_parameters, **optimizer_kwargs)
                if optimizer_cls.__name__ == "Adam8bit":
                    raise NotImplementedError

        if is_sagemaker_mp_enabled():
            raise NotImplementedError
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer

    def _get_local_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed

        # Build the sampler.
        if self.args.group_by_length:
            raise NotImplementedError
        else:
            return LocalDistributedSampler(
                self.train_dataset,
                num_replicas=self.args.world_size,
                rank=self.args.process_index,
                seed=seed,
            )

    def _get_local_train_sampler_aug_coyo(self) -> Optional[torch.utils.data.Sampler]:
        seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed

        # Build the sampler.
        if self.args.group_by_length:
            raise NotImplementedError
        else:
            sample_len_list = self.args.sample_lens
            return LocalDistributedSamplerAugCoyo(
                self.train_dataset,
                num_replicas=self.args.world_size,
                rank=self.args.process_index,
                seed=seed, batch_size=self.args.train_batch_size,
                sample_len_list=sample_len_list,
            )
