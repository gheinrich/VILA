# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import transformers
from transformers.image_transforms import (
    ChannelDimension,
    Iterable,
    Optional,
    Union,
    get_channel_dimension_axis,
    infer_channel_dimension_format,
    np,
    to_channel_dimension_format,
)


def patched_normalize(
    image: np.ndarray,
    mean: Union[float, Iterable[float]],
    std: Union[float, Iterable[float]],
    data_format: Optional[ChannelDimension] = None,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
) -> np.ndarray:
    """
    Normalizes `image` using the mean and standard deviation specified by `mean` and `std`.

    image = (image - mean) / std

    Args:
        image (`np.ndarray`):
            The image to normalize.
        mean (`float` or `Iterable[float]`):
            The mean to use for normalization.
        std (`float` or `Iterable[float]`):
            The standard deviation to use for normalization.
        data_format (`ChannelDimension`, *optional*):
            The channel dimension format of the output image. If unset, will use the inferred format from the input.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("image must be a numpy array")

    input_data_format = infer_channel_dimension_format(image)
    channel_axis = get_channel_dimension_axis(image)
    num_channels = image.shape[channel_axis]

    if isinstance(mean, Iterable):
        if len(mean) != num_channels:
            if num_channels == 1:
                num_channels = 3
                image = np.concatenate([image, image, image], axis=channel_axis)
            else:
                raise ValueError(f"mean must have {num_channels} elements if it is an iterable, got {len(mean)}")
    else:
        mean = [mean] * num_channels
    mean = np.array(mean, dtype=image.dtype)

    if isinstance(std, Iterable):
        if len(std) != num_channels:
            raise ValueError(f"std must have {num_channels} elements if it is an iterable, got {len(std)}")
    else:
        std = [std] * num_channels
    std = np.array(std, dtype=image.dtype)

    if input_data_format == ChannelDimension.LAST:
        image = (image - mean) / std
    else:
        image = ((image.T - mean) / std).T

    image = to_channel_dimension_format(image, data_format) if data_format is not None else image
    return image


def patch_normalize_preprocess():
    transformers.image_transforms.normalize = patched_normalize


import os

import torch
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.utils import logging

TRAINER_STATE_NAME = "trainer_state.json"
logger = logging.get_logger(__name__)


def _save_checkpoint(self, model, trial, metrics=None):
    # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
    # want to save except FullyShardedDDP.
    # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

    # Save model checkpoint
    checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

    if self.hp_search_backend is None and trial is None:
        self.store_flos()

    run_dir = self._get_output_dir(trial=trial)
    output_dir = os.path.join(run_dir, checkpoint_folder)

    if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
        logger.warning(
            f"Checkpoint destination directory {output_dir} already exists and is non-empty."
            "Saving will proceed but saved results may be invalid."
        )
        staging_output_dir = output_dir
    else:
        staging_output_dir = os.path.join(run_dir, f"tmp-{checkpoint_folder}")

    self.save_model(staging_output_dir, _internal_call=True)

    if not self.args.save_only_model:
        # Save optimizer and scheduler
        self._save_optimizer_and_scheduler(staging_output_dir)
        # Save RNG state
        self._save_rng_state(staging_output_dir)

    # Determine the new best metric / best model checkpoint
    if metrics is not None and self.args.metric_for_best_model is not None:
        metric_to_check = self.args.metric_for_best_model
        if not metric_to_check.startswith("eval_"):
            metric_to_check = f"eval_{metric_to_check}"
        metric_value = metrics[metric_to_check]

        operator = np.greater if self.args.greater_is_better else np.less
        if (
            self.state.best_metric is None
            or self.state.best_model_checkpoint is None
            or operator(metric_value, self.state.best_metric)
        ):
            self.state.best_metric = metric_value
            self.state.best_model_checkpoint = staging_output_dir

    # Save the Trainer state
    if self.args.should_save:
        self.state.save_to_json(os.path.join(staging_output_dir, TRAINER_STATE_NAME))

    if self.args.push_to_hub:
        self._push_from_checkpoint(staging_output_dir)

    torch.distributed.barrier()
    if staging_output_dir != output_dir:
        with self.args.main_process_first(
            desc="Renaming model checkpoint folder to true location", local=self.args.save_on_each_node
        ):
            if os.path.exists(staging_output_dir):
                os.rename(staging_output_dir, output_dir)

    # Maybe delete some older checkpoints.
    if self.args.should_save:
        # Solely rely on numerical checkpoint id for rotation.
        # mtime is not reliable especially on some fuse fs in cloud environments.
        self._rotate_checkpoints(use_mtime=False, output_dir=run_dir)
