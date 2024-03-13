from PIL import Image
import os
import json
import os, os.path as osp
import json

from torch.utils.data import Dataset

DEFAULT_TEXTOCR = "~/nvr_elm_llm/dataset/TextOCR"
DEFAULT_TEXTOCR = osp.expanduser(DEFAULT_TEXTOCR)

class GenericDataset:
    def add(self, dataset):
        return SummedDataset(self, dataset)

    def resize_image(self, image):
        original_width, original_height = image.size

        original_height = max(original_height, 1)
        original_width = max(original_width, 1)

        scale = self.image_height / original_height

        resized_width = int(round(scale * original_width, 0))
        new_width = resized_width + (
            self.patch_width - (resized_width % self.patch_width)
        )  # Adjusted this line

        return image.resize((new_width, self.image_height))

    def __add__(self, dataset):
        return self.add(dataset)


class SummedDataset(GenericDataset):
    def __init__(self, dataset_left, dataset_right) -> None:
        self.left = dataset_left
        self.right = dataset_right

    def __len__(self):
        return len(self.left) + len(self.right)

    def __getitem__(self, idx):

        if idx > (len(self.left) - 1):
            idx_corrected = idx % len(self.left)
            return self.right[idx_corrected]

        return self.left[idx]


class TextOCRDataset(GenericDataset):
    name = "text_ocr_dataset"
    def __init__(
        self,
        base_folder=DEFAULT_TEXTOCR,
        split="train",
        transforms=lambda x: x,
        min_area=0.001,
    ) -> None:
        super().__init__()

        self.split = split
        self.transforms = transforms
        self.data = []
        self.img2text = {}

        annotations = json.load(
            open(os.path.join(base_folder, f"TextOCR_0.1_{split}.json"), "r")
        )
        valid_images = [
            {
                "size": (annotations["imgs"][img]["width"], annotations["imgs"][img]["height"]),
                "path": os.path.join(
                    base_folder,
                    annotations["imgs"][img]["file_name"].replace("train/", "train_images/"),
                ),
                "annots": [str(i) for i in annotations["imgToAnns"][img]],
            }
            for img in annotations["imgs"]
        ]

        for image in valid_images:
            for ann in image["annots"]:
                annotation = annotations["anns"][ann]
                if annotation["utf8_string"] == ".":
                    continue  # Unreadable characters
                
                x,y,w,h = [int(x) for x in annotation["bbox"]]
                img_area = image["size"][0] * image["size"][1]
                if (w * h) / img_area < min_area:
                    continue # skip too small texts
                
                fpath = image["path"]
                self.data.append(
                    {
                        "image_path": fpath,
                        "bbx": [int(x) for x in annotation["bbox"]],
                        "transcription": annotation["utf8_string"],
                    }
                )
                
                if fpath not in self.img2text:
                    self.img2text[fpath] = []
                self.img2text[fpath].append({
                    "bbx": [int(x) for x in annotation["bbox"]],
                    "transcription": annotation["utf8_string"],
                })
                
        self.image_ids = list(self.img2text.keys())

    def __len__(self):
        # return len(self.data)
        return len( self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        metadata = self.img2text[img_id]
        origin_image = Image.open(img_id)
        
        annotation = [_["transcription"] for _ in metadata]
        bboxes = [_["bbx"] for _ in metadata]
        
        return {
            "image_path": img_id,
            "origin_image": origin_image,
            "annotation": annotation,
            "bboxes": bboxes,
            "dataset": self.name,
            "split": self.split,
        }
        

    def __getitem__single__(self, idx):
        metadata = self.data[idx]
        origin_image = Image.open(metadata["image_path"])
        x, y, w, h = metadata["bbx"]
        cropped_image = (
            Image.open(metadata["image_path"]).crop((x, y, x + w, y + h)).convert("RGB")
        )
        return {
            "image_path": metadata["image_path"],
            "origin_image": origin_image,
            "cropped_image": cropped_image,
            "annotation": metadata["transcription"],
            "dataset": self.name,
            "split": self.split,
            "tokens": [char for char in metadata["transcription"]],
        }



class VILAOCRDataset(Dataset):
    def __init__(self, 
            base_folder="~/nvr_elm_llm/dataset/TextOCR",
            split="train",
            min_area=0.001) -> None:
        super().__init__()
        
        base_folder = osp.expanduser(base_folder)
        self.dataset = TextOCRDataset(base_folder, split, min_area=min_area)
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        meta =  self.dataset[index]
        
        img = meta["origin_image"]
        fpath = meta["image_path"]
        text = " ".join(meta["annotation"])
        
        prompt = f"Please read the text on image and type it below, each word separated by space.\n{text}"
        
        return {
            "image": img,
            "fpath": fpath,
            "text": prompt,
        }
        
        


if __name__ == "__main__":
    from pprint import pprint
    # dataset = TextOCRDataset()
    dataset = VILAOCRDataset()

    for idx in range(5):
        pprint(dataset[idx])
