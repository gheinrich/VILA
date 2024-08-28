import os
from typing import Any, Optional

from hydra.utils import instantiate
from torch.utils.data import ConcatDataset, Dataset
from transformers import PreTrainedTokenizer

from llava.data.datasets_mixture import DATASETS_LEGACY
from llava.train.args import DataArguments, TrainingArguments
from llava.utils import io
from llava.utils.logging import logger

__all__ = ["DATASETS", "register_datasets", "build_dataset"]


def register_datasets(name: Optional[str] = None):
    if name is None:
        name = os.environ.get("VILA_DATASETS", "default")
        logger.info(f"Registering datasets from `{name}`.")
    return io.load(os.path.join(os.path.dirname(__file__), "registry", f"{name}.yaml"))


DATASETS = register_datasets()


class RepeatedDataset(Dataset):
    def __init__(self, dataset: Dataset, times: int) -> None:
        super().__init__()
        self.dataset = dataset
        self.times = times

    def __len__(self) -> int:
        return len(self.dataset) * self.times

    def __getitem__(self, index: int) -> Any:
        return self.dataset[index % len(self.dataset)]


def build_dataset(
    mixture: str,
    data_args: DataArguments,
    training_args: TrainingArguments,
    tokenizer: PreTrainedTokenizer,
) -> Dataset:
    datasets = []
    for name in mixture.strip().lower().split("+"):
        if "*" in name:
            name, times = name.split("*")
            times = int(times)
        else:
            times = 1

        if DATASETS is not None and name in DATASETS:
            dataset = instantiate(DATASETS[name], _partial_=True)(
                tokenizer=tokenizer,
                data_args=data_args,
            )
        elif name in DATASETS_LEGACY:
            logger.warning(f"Dataset {name} is registered under legacy mode.")
            dataset = build_dataset_legacy(
                name,
                data_args=data_args,
                training_args=training_args,
                tokenizer=tokenizer,
            )
        else:
            raise ValueError(f"Dataset {name} is not registered.")

        if times > 1:
            dataset = RepeatedDataset(dataset, times)
        datasets.append(dataset)
    return ConcatDataset(datasets)


def build_dataset_legacy(
    name: str,
    data_args: DataArguments,
    training_args: TrainingArguments,
    tokenizer: PreTrainedTokenizer,
) -> Dataset:
    from llava.data.dataset import (
        DummyDataset,
        LazyCCSWebDataset,
        LazyCoyoDataset,
        LazyCoyoWebDataset,
        LazyMMC4Dataset,
        LazySupervisedDataset,
        LazyVFlanDataset,
        LazyVideoWebDataset,
        LazyWDSDataset,
        VILAPanda70m_LongSeq,
    )
    from llava.data.dataset_impl.coyo_recap import LazyCoyoWebRecapDataset
    from llava.data.dataset_impl.general_img_text import LazyImageTextWebDataset
    from llava.data.dataset_impl.hiertext import VILAHierText
    from llava.data.dataset_impl.panda70m import VILAPanda70m
    from llava.data.dataset_impl.sam import LazySAMWebDataset
    from llava.data.dataset_impl.textocr import VILATextOCR

    dataset = DATASETS_LEGACY[name]
    dataset_type = dataset.dataset_type
    if dataset_type == "torch":
        dataset_cls = LazySupervisedDataset
    elif dataset_type == "wds":
        dataset_cls = LazyWDSDataset
    elif dataset_type == "mmc4":
        dataset_cls = LazyMMC4Dataset
    elif dataset_type == "coyo":
        dataset_cls = LazyCoyoDataset
    elif dataset_type == "sam-wds":
        dataset_cls = LazySAMWebDataset
    elif dataset_type == "coyo-wds":
        dataset_cls = LazyCoyoWebDataset
    elif dataset_type == "coyo-wds-recap":
        dataset_cls = LazyCoyoWebRecapDataset
    elif dataset_type == "textocr":
        dataset_cls = VILATextOCR
    elif dataset_type == "hiertext":
        dataset_cls = VILAHierText
    elif dataset_type == "panda70m":
        dataset_cls = VILAPanda70m
    elif dataset_type == "ccs-wds":
        dataset_cls = LazyCCSWebDataset
    elif dataset_type == "vflan":
        dataset_cls = LazyVFlanDataset
    elif dataset_type == "video-wds":
        dataset_cls = LazyVideoWebDataset
    elif dataset_type == "imgtxt-wds":
        dataset_cls = LazyImageTextWebDataset
    elif dataset_type == "evaluation":
        dataset_cls = LazyEvaluateDataset
    elif dataset_type == "dummy":
        dataset_cls = DummyDataset
    elif dataset_type == "panda70m_sp":
        dataset_cls = VILAPanda70m_LongSeq
    else:
        raise NotImplementedError(f"{dataset_type} is not supported.")

    data_args.meta_path = getattr(dataset, "meta_path", None)
    data_args.caption_choice = getattr(dataset, "caption_choice", None)
    data_args.caption_choice_2 = getattr(dataset, "caption_choice_2", None)
    data_args.start_idx = getattr(dataset, "start_idx", None)
    data_args.end_idx = getattr(dataset, "end_idx", None)

    return dataset_cls(
        tokenizer=tokenizer,
        data_path=dataset.data_path,
        image_folder=getattr(dataset, "image_path"),
        data_args=data_args,
        training_args=training_args,
    )
