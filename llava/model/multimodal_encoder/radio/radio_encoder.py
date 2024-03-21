import warnings
from argparse import Namespace
from typing import Any, Dict

import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer
from transformers import CLIPImageProcessor, SamImageProcessor

from .create_model import create_model_from_args
from .enable_spectral_reparam import configure_spectral_reparam_from_args
from .token_merging import kth_bipartite_soft_matching


class RunningAverage:
    def __init__(self):
        self.total = 0
        self.count = 0

    def add_value(self, value):
        self.total += value
        self.count += 1
        return self.total / self.count


def get_prefix_state_dict(state_dict: Dict[str, Any], prefix: str):
    mod_state_dict = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}
    return mod_state_dict


class SetDefaultDtypeContext:
    def __init__(self, dtype):
        self.dtype = dtype
        self.previous_dtype = torch.get_default_dtype()

    def __enter__(self):
        torch.set_default_dtype(self.dtype)

    def __exit__(self, exc_type, exc_value, traceback):
        torch.set_default_dtype(self.previous_dtype)


class RADIOEncoder(nn.Module):
    def __init__(self, config, delay_load=False):
        super().__init__()

        self.is_loaded = False

        print(f"RADIO model config: {config}")

        vision_tower = config.vision_tower

        # The vision_tower_name is expected in the format of:
        # "radio:<image_size>:<vision_tower_checkpoint>", where <image_size>
        # is the size of the input image and <vision_tower_checkpoint> is the
        # path to the checkpoint file.

        # Remove the "radio:" prefix from the vision tower name.
        self.vision_tower_name = vision_tower[len("radio:") :]
        self.image_size = int(self.vision_tower_name.split(":")[0])
        self.vision_tower_checkpoint = self.vision_tower_name.split(":")[1]

        self.select_feature = getattr(config, "mm_vision_select_feature", "patch")
        self.activation_multiplier = getattr(config, "mm_vision_activation_multiplier", 1.0)
        self.do_center_crop = True

        crop_size = {"height": self.image_size, "width": self.image_size}
        if self.do_center_crop:
            self.image_processor = CLIPImageProcessor(
                size={"shortest_edge": self.image_size},
                crop_size=crop_size,
                do_center_crop=self.do_center_crop,
                do_normalize=True,
            )
        else:
            self.image_processor = SamImageProcessor(
                size={"longest_edge": self.image_size},
                pad_size={"height": self.image_size, "width": self.image_size},
                do_pad=False,
                do_normalize=True,
            )
            # Add a crop_size attribute to the image processor, since the
            # train.py script needs this to generate fake images of zeros
            # with the right size, when the sample does not have an
            # associated image.
            self.image_processor.crop_size = crop_size

        print("RADIO Encoder image processor:", self.image_processor)

        self.load_config()

        if delay_load:
            self.is_loaded = False
        else:
            self.load_model()

        self.average_tokens = RunningAverage()

    def load_config(self):
        # Load weights from checkpoint.
        checkpoint_path = self.vision_tower_checkpoint
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        with SetDefaultDtypeContext(torch.float32):

            if "args" in checkpoint:
                self._args = checkpoint["args"]
                self.base_model = create_model_from_args(self._args)

                if "state_dict_ema" in checkpoint:
                    self._pretrained_state_dict = checkpoint["state_dict_ema"]
                    # Disable specteal reparamaterization if we are loading the EMA model.
                    checkpoint["args"].spectral_reparam = False
                elif "state_dict" in checkpoint:
                    self._pretrained_state_dict = checkpoint["state_dict"]
            else:
                timm_model_name = self.vision_tower_name[len("timm/") :]
                self._args = Namespace(model=timm_model_name, pretrained=False)
                self.base_model = create_model_from_args(self._args)

                self._pretrained_state_dict = checkpoint

        if isinstance(self.base_model, VisionTransformer):
            self.hidden_size = self.base_model.embed_dim
        else:
            raise ValueError(f"Unknown model type: {self.base_model}")

        if hasattr(self.base_model, "patch_generator"):
            patch_gen = self.base_model.patch_generator
            # Cropped Positional Embedding (CPE) case.
            self.patch_size = patch_gen.patch_size
        else:
            # Standard ViT case.
            self.patch_size = self.base_model.patch_embed.patch_size

    def load_model(self):

        self.base_model.float()

        configure_spectral_reparam_from_args(self.base_model, self._args)

        checkpoint_state_dict = self._pretrained_state_dict

        # Remove the 'base_model' prefix from all keys in the state dict.
        checkpoint_state_dict = get_prefix_state_dict(checkpoint_state_dict, "base_model.")

        model_state_dict = self.base_model.state_dict()

        print("checkpoint keys", checkpoint_state_dict.keys())

        for k, v in checkpoint_state_dict.items():
            if k not in model_state_dict:
                print(f"Checkpoint key {k} not in model state dict")
            elif model_state_dict[k].shape != v.shape:
                print(
                    f"Checkpoint key {k} has shape {v.shape} but model state dict has shape {model_state_dict[k].shape}"
                )

        # Only keep keys whose size matches that of the model parameters.
        checkpoint_state_dict = {
            k: v
            for k, v in checkpoint_state_dict.items()
            if k in model_state_dict and model_state_dict[k].shape == v.shape
        }
        self._pretrained_state_dict = checkpoint_state_dict

        print("keys to restore", checkpoint_state_dict.keys())

        print(self.base_model.load_state_dict(checkpoint_state_dict, strict=False))

        for name, param in self.base_model.named_parameters():
            if not torch.all(torch.isfinite(param)):
                print(f"Post load Parameter {name} has non-finite values.")

        self.base_model.eval()
        self.base_model.requires_grad_(False)

        self.is_loaded = True

    def train(self, mode=True):
        """Intercept call."""
        # Drop a warning if mode is True.
        if mode:
            warnings.warn("RADIOEncoder is always in eval mode.")
        pass

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        input_dtype = x.dtype

        # Convert the input to the model's dtype (we assume
        # that the model only has one dtype for all parameters).

        # param0 = next(self.base_model.parameters())

        # if param0.dtype != torch.float32:
        #    print(f"Converting model to float32")
        #    self.base_model = self.base_model.to(dtype=torch.float32)
        #    self.base_model.load_state_dict(self._pretrained_state_dict, strict=True)

        # x = x.to(dtype=torch.float32, device=param0.device)

        # Add a batch dimension if necessary.
        x_shape = x.shape
        if len(x_shape) == 3:
            x = x.unsqueeze(0)

        multiple = self.patch_size
        if self.do_center_crop:
            # Crop the input to a multiple of patch size.
            _, _, H, W = x.shape

            H = H - (H % multiple)
            W = W - (W % multiple)

            # H = min(H, 700)
            # W = min(W, 700)
            x = x[:, :, :H, :W]
        else:
            # Pad to nearest multiple of patch size
            _, _, H, W = x.shape
            H = H + (multiple - (H % multiple)) % multiple
            W = W + (multiple - (W % multiple)) % multiple
            x = nn.functional.pad(x, (0, W - x.shape[3], 0, H - x.shape[2]), mode="constant", value=0)

        print(f"input shape={x_shape}->{x.shape} device={x.device} mean={x.mean().item()} std={x.std().item()}")

        # Extract features using the base_model.
        if isinstance(self.base_model, VisionTransformer):
            if hasattr(self.base_model, "patch_generator"):
                num_cls_tokens = self.base_model.patch_generator.num_cls_tokens
                num_registers = self.base_model.patch_generator.num_registers
                num_summary_tokens = num_cls_tokens + num_registers
            else:
                num_summary_tokens = 1
                num_cls_tokens = 1
            features = self.base_model.forward_features(x)
            if self.select_feature == "patch":
                # Remove summary tokens.
                features = features[:, num_summary_tokens:]
            elif self.select_feature == "cls_patch":
                cls_tokens = features[:, :num_cls_tokens]
                spatial_tokens = features[:, num_summary_tokens:]
                features = torch.cat([cls_tokens, spatial_tokens], dim=1)
            elif self.select_feature == "cls":
                # Remove patch tokens.
                features = features[:, :num_summary_tokens]
            elif self.select_feature == "random":
                # Retrieve 8 random tokens out of the original sequence.
                num_tokens = features.shape[1]
                indices = torch.randperm(num_tokens)[:8]
                features = features[:, indices]
            elif self.select_feature == "random_spatial":
                # Retrieve 8 random tokens out of the spatial tokens.
                num_tokens = features.shape[1] - num_summary_tokens
                indices = torch.randperm(num_tokens)[:8] + num_summary_tokens
                features = features[:, indices]
            elif self.select_feature == "summary_random_spatial":
                summary_tokens = features[:, :num_summary_tokens]
                num_spatial_tokens = features.shape[1] - num_summary_tokens
                num_spatial_tokens_to_retain = num_spatial_tokens // 2
                indices = torch.randperm(num_spatial_tokens)[:num_spatial_tokens_to_retain] + num_summary_tokens
                spatial_tokens = features[:, indices]
                features = torch.cat([summary_tokens, spatial_tokens], dim=1)
            elif self.select_feature == "summary_merge_spatial":
                summary_tokens = features[:, :num_summary_tokens]
                spatial_tokens = features[:, num_summary_tokens:]
                merge, _ = kth_bipartite_soft_matching(spatial_tokens, k=2)
                merged_spatial_tokens = merge(spatial_tokens)
                features = torch.cat([summary_tokens, merged_spatial_tokens], dim=1)
            else:
                raise ValueError(f"Unexpected select feature: {self.select_feature}")

            num_tokens = features.shape[1]
            average_tokens = self.average_tokens.add_value(num_tokens)
            print(f"Average number of tokens: {average_tokens:.2f}")
        else:
            raise ValueError("Unhandled model")

        # Remove the batch dimension if we added it.
        if len(x_shape) == 3:
            features = features.squeeze(0)

        features = features * self.activation_multiplier

        print(
            f"features shape={features.shape} device={features.device} mean={features.mean().item()} std={features.std().item()}"
        )

        return features.to(dtype=input_dtype)
