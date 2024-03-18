# refernced from https://github.com/CVC-DAG/OCR_datasets/blob/master/src/datasets/ocr/hiertext.py
from PIL import Image
import os
import json
from collections import defaultdict
from llava.data.dataset_impl.textocr import GenericDataset

DEFAULT_HIERTEXT = "/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/dataset/hiertext"


def bbx_from_vertices_list(vertices):
    # Care about the index, potential source of errors
    return (
        min(vertices, key=lambda x: x[0])[0],
        min(vertices, key=lambda x: x[1])[1],
        max(vertices, key=lambda x: x[0])[0],
        max(vertices, key=lambda x: x[1])[1],
    )


class HierTextDataset(GenericDataset):
    name = "hiertext_dataset"

    def __init__(
        self,
        base_folder=DEFAULT_HIERTEXT,
        split="train",
        handwritten=[True, False],
        legibility=[True, False],
        mode="words",
        image_height=128,
        patch_width=16,
        transforms=lambda x: x,
    ) -> None:
        self.image_height = image_height
        self.patch_width = patch_width
        self.transforms = transforms
        self.split = f"{split}_legibility-{legibility}_handwritten-{handwritten}"

        annotation_file = json.load(
            open(
                os.path.join(
                    base_folder,
                    "gt",
                    "train.jsonl" if split == "train" else "validation.jsonl",
                ),
                "r",
            )
        )
        images_path = os.path.join(
            base_folder, "train" if split == "train" else "validation"
        )
        self.base_images = images_path

        self.samples = []
        self.unique_fpath = set()
        self.unique_samples = defaultdict(list)
        for num, annotation in enumerate(annotation_file["annotations"]):
            image_path = os.path.join(images_path, annotation["image_id"] + ".jpg")
            for paragraph in annotation["paragraphs"]:
                for line in paragraph["lines"]:
                    x, y, x2, y2 = bbx_from_vertices_list(line["vertices"])

                    if x2 * y2 < 100:
                        # skip too small texts
                        continue
                    if x2 - x < y2 - y:
                        continue  # TODO: Evaluation without vertical lines. Not fair.
                    if (
                        line["legible"] in legibility
                        and line["handwritten"] in handwritten
                        and not line["vertical"]
                    ):
                        if mode == "lines":
                            data = {
                                "bbx": bbx_from_vertices_list(line["vertices"]),
                                "image_path": image_path,
                                "transcription": line["text"],
                                "vertical": line["vertical"],
                            }
                            self.samples.append(data)
                            self.unique_samples[image_path].append(data)
                            self.unique_fpath.add(image_path)
                        else:
                            for word in line["words"]:
                                if not word["vertical"]:
                                    data = {
                                        "bbx": bbx_from_vertices_list(word["vertices"]),
                                        "image_path": image_path,
                                        "transcription": word["text"],
                                        "vertical": word["vertical"],
                                    }
                                    self.samples.append(data)
                                    self.unique_samples[image_path].append(data)
                                    self.unique_fpath.add(image_path)
        self.unique_fpath = list(self.unique_fpath)

    def __len__(self):
        return len(self.unique_fpath)

    def __getitem__(self, idx):
        # metadata = self.samples[idx]
        # img_path = os.path.join(self.base_images, metadata["image_path"])
        # image = Image.open(img_path).convert("RGB")
        
        img_path = self.unique_fpath[idx]
        metadatas = self.unique_samples[img_path]

        annotations = []
        for metadata in metadatas:
            annotations.append(
                metadata["transcription"]
            )
        image = Image.open(img_path).convert("RGB")
            
        return {
            "img_path": img_path,
            "original_image": image,
            "annotation": annotations,
            "dataset": self.name,
            "split": self.split,
        }


if __name__ == "__main__":
    dst = HierTextDataset()
    for i in range(3):
        print(dst[i])
