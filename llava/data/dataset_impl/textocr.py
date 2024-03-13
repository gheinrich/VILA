import os, os.path as osp
import json
data_path = "~/nvr_elm_llm/dataset/TextOCR/TextOCR_0.1_train.json"
data_path = osp.expanduser(data_path)

"~/nvr_elm_llm/dataset/TextOCR/train_images/6029c75e0325d164.jpg"

info = json.load(open(data_path, "r"))
# In [30]: info.keys()
# Out[30]: dict_keys(['info', 'imgs', 'anns', 'imgToAnns'])

img_info = info["imgs"]
imgToAnns_info = info["imgToAnns"]
key = "6029c75e0325d164"
imgToAnns_info[key]

print(img_info[key])
for k in imgToAnns_info[key]:
    res = info["anns"][k]
    print(f'''bbox: {res["bbox"]}\ntext: {res["utf8_string"]}\n---''')
    