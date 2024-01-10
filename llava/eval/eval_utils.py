import torch
import os

from PIL import Image
import requests
from io import BytesIO
from torch.utils.data import Dataset
import random
import json

import numpy as np



def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def prepare_eval_samples(test_dataset, num_samples, seed):
    if num_samples < len(test_dataset):  # otherwise do not sample
        np.random.seed(seed)
        random_indices = np.random.choice(
            len(test_dataset), num_samples, replace=False)
        return torch.utils.data.Subset(test_dataset, random_indices)
    else:
        return test_dataset


def get_query_set(train_dataset, query_set_size, seed):
    # subsample the training set for few-shot experiments
    np.random.seed(seed)
    query_set = np.random.choice(
        len(train_dataset), query_set_size, replace=False)
    return [train_dataset[i] for i in query_set]


def sample_batch_demos_from_query_set(query_set, num_samples, batch_size):
    return [random.sample(query_set, num_samples) for _ in range(batch_size)]


class NoCapDataset(Dataset):
    def __init__(self, image_path, json_path):
        with open(json_path) as f:
            self.annotations = json.load(f)
        self.image_path = image_path
    
    def __len__(self):
        return len(self.annotations["annotations"])

    def __getitem__(self, idx):
        anno = self.annotations["annotations"][idx]
        caption = anno["caption"]
        image_id = anno["image_id"]
        image_info = self.annotations["images"][image_id]
        assert image_info["id"] == image_id
        img_path = image_info["file_name"]
        img_path = os.path.join(self.image_path, img_path)

        return {
                "image": load_image(img_path),
                "caption": caption,
                "image_id": image_id,
            }


class CaptionDataset(Dataset):
    def __init__(
        self,
        image_train_dir_path,
        annotations_path,
        is_train,
        dataset_name,
        image_val_dir_path=None,
    ):
        self.image_train_dir_path = image_train_dir_path
        self.image_val_dir_path = image_val_dir_path
        self.annotations = []
        self.is_train = is_train
        self.dataset_name = dataset_name

        full_annotations = json.load(open(annotations_path))["images"]

        for i in range(len(full_annotations)):
            if self.is_train and full_annotations[i]["split"] != "train":
                continue
            elif not self.is_train and full_annotations[i]["split"] != "test":
                continue

            self.annotations.append(full_annotations[i])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if self.dataset_name == "coco":
            img_path = os.path.join(
                self.image_train_dir_path, self.annotations[idx]["filename"].replace(
                    "COCO_train2014_", "")
            ) if self.annotations[idx]["filepath"] == "train2014" \
                else os.path.join(
                    self.image_val_dir_path, self.annotations[idx]["filename"].replace(
                        "COCO_train2014_", "")
            )

        elif self.dataset_name == "flickr":
            img_path = os.path.join(
                self.image_train_dir_path, self.annotations[idx]["filename"]
            )
        caption = self.annotations[idx]["sentences"][0]["raw"]
        return {
            "image": load_image(img_path),
            "caption": caption,
            "image_id": self.annotations[idx]["cocoid"]
            if self.dataset_name == "coco"
            else self.annotations[idx]["filename"].split(".")[0],
        }


class VQADataset(Dataset):
    def __init__(
        self, image_dir_path, question_path, annotations_path, is_train, dataset_name
    ):
        self.questions = json.load(open(question_path, "r"))["questions"]
        self.answers = json.load(open(annotations_path, "r"))["annotations"]
        self.image_dir_path = image_dir_path
        self.is_train = is_train
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.questions)

    def get_img_path(self, question):
        if self.dataset_name in ["vqav2", "okvqa"]:
            return os.path.join(
                self.image_dir_path,
                f"COCO_train2014_{question['image_id']:012d}.jpg"
                if self.is_train
                else f"COCO_val2014_{question['image_id']:012d}.jpg",
            )
        elif self.dataset_name == "vizwiz":
            return os.path.join(self.image_dir_path, question["image_id"])
        elif self.dataset_name == "textvqa":
            return os.path.join(self.image_dir_path, f"{question['image_id']}.jpg")
        else:
            raise Exception(f"Unknown VQA dataset {self.dataset_name}")

    def __getitem__(self, idx):
        question = self.questions[idx]
        answers = self.answers[idx]
        img_path = self.get_img_path(question)
        image = Image.open(img_path)
        image.load()
        return {
            "image": image,
            "question": question["question"],
            "answers": [a["answer"] for a in answers["answers"]],
            "question_id": question["question_id"],
            "image_id": question["image_id"],
        }


class HatefulMemesDataset(Dataset):
    def __init__(self, image_dir_path, annotations_path):
        self.image_dir_path = image_dir_path
        with open(annotations_path, "r") as f:
            self.annotations = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        img_path = os.path.join(self.image_dir_path, annotation["img"].split("/")[-1])
        image = Image.open(img_path)
        image.load()
        return {
            "id": annotation["id"],
            "image": image,
            "ocr": annotation["text"],
            "class_name": "yes" if annotation["label"] == 1 else "no",
            "class_id": annotation["label"],
        }


# Those are manual mapping that are not caught by our stemming rules or would
# would be done incorrectly by our automatic stemming rule. In details,
# the keys of the _MANUAL_MATCHES dict contains the original word and the value
# contains the transformation of the word expected by the OKVQA stemming rule.
# These manual rules were found by checking the `raw_answers` and the `answers`
# fields of the released OKVQA dataset and checking all things that were not
# properly mapped by our automatic rules. In particular some of the mapping
# are sometimes constant, e.g. christmas -> christmas which was incorrectly
# singularized by our inflection.singularize.
import re
import nltk
from nltk.corpus.reader import VERB
import inflection

_MANUAL_MATCHES = {
    "police": "police",
    "las": "las",
    "vegas": "vegas",
    "yes": "yes",
    "jeans": "jean",
    "hell's": "hell",
    "domino's": "domino",
    "morning": "morn",
    "clothes": "cloth",
    "are": "are",
    "riding": "ride",
    "leaves": "leaf",
    "dangerous": "danger",
    "clothing": "cloth",
    "texting": "text",
    "kiting": "kite",
    "firefighters": "firefight",
    "ties": "tie",
    "married": "married",
    "teething": "teeth",
    "gloves": "glove",
    "tennis": "tennis",
    "dining": "dine",
    "directions": "direct",
    "waves": "wave",
    "christmas": "christmas",
    "drives": "drive",
    "pudding": "pud",
    "coding": "code",
    "plating": "plate",
    "quantas": "quanta",
    "hornes": "horn",
    "graves": "grave",
    "mating": "mate",
    "paned": "pane",
    "alertness": "alert",
    "sunbathing": "sunbath",
    "tenning": "ten",
    "wetness": "wet",
    "urinating": "urine",
    "sickness": "sick",
    "braves": "brave",
    "firefighting": "firefight",
    "lenses": "lens",
    "reflections": "reflect",
    "backpackers": "backpack",
    "eatting": "eat",
    "designers": "design",
    "curiousity": "curious",
    "playfulness": "play",
    "blindness": "blind",
    "hawke": "hawk",
    "tomatoe": "tomato",
    "rodeoing": "rodeo",
    "brightness": "bright",
    "circuses": "circus",
    "skateboarders": "skateboard",
    "staring": "stare",
    "electronics": "electron",
    "electicity": "elect",
    "mountainous": "mountain",
    "socializing": "social",
    "hamburgers": "hamburg",
    "caves": "cave",
    "transitions": "transit",
    "wading": "wade",
    "creame": "cream",
    "toileting": "toilet",
    "sautee": "saute",
    "buildings": "build",
    "belongings": "belong",
    "stockings": "stock",
    "walle": "wall",
    "cumulis": "cumuli",
    "travelers": "travel",
    "conducter": "conduct",
    "browsing": "brows",
    "pooping": "poop",
    "haircutting": "haircut",
    "toppings": "top",
    "hearding": "heard",
    "sunblocker": "sunblock",
    "bases": "base",
    "markings": "mark",
    "mopeds": "mope",
    "kindergartener": "kindergarten",
    "pies": "pie",
    "scrapbooking": "scrapbook",
    "couponing": "coupon",
    "meetings": "meet",
    "elevators": "elev",
    "lowes": "low",
    "men's": "men",
    "childrens": "children",
    "shelves": "shelve",
    "paintings": "paint",
    "raines": "rain",
    "paring": "pare",
    "expressions": "express",
    "routes": "rout",
    "pease": "peas",
    "vastness": "vast",
    "awning": "awn",
    "boy's": "boy",
    "drunkenness": "drunken",
    "teasing": "teas",
    "conferences": "confer",
    "ripeness": "ripe",
    "suspenders": "suspend",
    "earnings": "earn",
    "reporters": "report",
    "kid's": "kid",
    "containers": "contain",
    "corgie": "corgi",
    "porche": "porch",
    "microwaves": "microwave",
    "batter's": "batter",
    "sadness": "sad",
    "apartments": "apart",
    "oxygenize": "oxygen",
    "striping": "stripe",
    "purring": "pure",
    "professionals": "profession",
    "piping": "pipe",
    "farmer's": "farmer",
    "potatoe": "potato",
    "emirates": "emir",
    "womens": "women",
    "veteran's": "veteran",
    "wilderness": "wilder",
    "propellers": "propel",
    "alpes": "alp",
    "charioteering": "chariot",
    "swining": "swine",
    "illness": "ill",
    "crepte": "crept",
    "adhesives": "adhesive",
    "regent's": "regent",
    "decorations": "decor",
    "rabbies": "rabbi",
    "overseas": "oversea",
    "travellers": "travel",
    "casings": "case",
    "smugness": "smug",
    "doves": "dove",
    "nationals": "nation",
    "mustange": "mustang",
    "ringe": "ring",
    "gondoliere": "gondolier",
    "vacationing": "vacate",
    "reminders": "remind",
    "baldness": "bald",
    "settings": "set",
    "glaced": "glace",
    "coniferous": "conifer",
    "revelations": "revel",
    "personals": "person",
    "daughter's": "daughter",
    "badness": "bad",
    "projections": "project",
    "polarizing": "polar",
    "vandalizers": "vandal",
    "minerals": "miner",
    "protesters": "protest",
    "controllers": "control",
    "weddings": "wed",
    "sometimes": "sometime",
    "earing": "ear",
}


class OKVQAStemmer:
    """Stemmer to match OKVQA v1.1 procedure."""

    def __init__(self):
        self._wordnet_lemmatizer = nltk.stem.WordNetLemmatizer()

    def stem(self, input_string):
        """Apply stemming."""
        word_and_pos = nltk.pos_tag(nltk.tokenize.word_tokenize(input_string))
        stemmed_words = []
        for w, p in word_and_pos:
            if w in _MANUAL_MATCHES:
                w = _MANUAL_MATCHES[w]
            elif w.endswith("ing"):
                w = self._wordnet_lemmatizer.lemmatize(w, VERB)
            elif p.startswith("NNS") or p.startswith("NNPS"):
                w = inflection.singularize(w)
            stemmed_words.append(w)
        return " ".join(stemmed_words)


stemmer = OKVQAStemmer()


def postprocess_ok_vqa_generation(predictions) -> str:
    prediction = re.split("Question|Answer|Short", predictions, 1)[0]
    prediction_stem = stemmer.stem(prediction)
    return prediction_stem


def postprocess_vqa_generation(predictions):
    answer = re.split("Question|Answer|Short", predictions, 1)[0]
    answer = re.split(", ", answer, 1)[0]
    return answer
