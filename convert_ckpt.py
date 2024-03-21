import json
import os
import os.path as osp
import shutil
import sys


def main(ckpt="checkpoints/vila-7B", overwrite=False):
    ckpt = osp.expanduser(ckpt)
    config_fpath = osp.join(ckpt, "config.json")
    shutil.copy(config_fpath, osp.join(ckpt, "config_bak.json"))
    cfg = json.load(open(config_fpath, "r"))

    if "attention_dropout" in cfg or "rope_theta" in cfg or "attention_bias" in cfg:

        if not overwrite:
            print("Skip as the ckpt is up-to-date, add --overwrite if you still want to covnert.")
            return

    cfg["attention_dropout"] = 0.0
    cfg["rope_theta"] = 10000.0
    cfg["attention_bias"] = False

    print("Appending attributes done.")

    json.dump(cfg, open(config_fpath, "w"))


if __name__ == "__main__":
    import fire

    fire.Fire(main)
