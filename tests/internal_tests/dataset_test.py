# Test by RANK=0 WORLD_SIZE=8 python /lustre/fsw/portfolios/llmservice/users/jasonlu/workspace/multi-modality-research/VILA/llava/train/dataset_test.py

from llava import conversation as conversation_lib
from llava.train import arguments
from llava.train import dataset
from llava.train import datasets_mixture
import transformers
import torch
import pprint
from llava.train.token_config import (
    DEFAULT_IMAGE_PATCH_TOKEN,
)
import json
import pickle
import sys
from pytorchvideo.data.encoded_video import EncodedVideo


video_path = "/home/jasonlu/video_datasets/Video_ChatGPT/VideoInstruct-100K/VideoInstruct100K.json"
video_path_out = "/home/jasonlu/video_datasets/Video_ChatGPT/VideoInstruct-100K/filtered_VideoInstruct100K.json"


def test_valid_video():
    video_json = json.load(open(video_path, "r"))
    video_json_output = []
    for video in video_json:
        path = "/home/jasonlu/video_datasets/Video_ChatGPT/activitynet_videos/" + video["video_id"] + ".mp4"
        try:
            video_data = EncodedVideo.from_path(path, decoder="decord", decode_audio=False)
            del video_data
        except:
            print(path)
            continue
        video_json_output.append(video)
    out_file = open(video_path_out, "w")
    json.dump(video_json_output, out_file)
    video_path.close()
    video_json_output.close()


test_valid_video()

# test_pkl_path = '/home/jil/datasets/vlm-flan-clean-text1m-nosqa/text_flan_1m.pkl'

# def test_text_only():
#     list_data_dict = []
#     with open(test_pkl_path, "rb") as f:
#         data = pickle.load(f)
#         list_data_dict.extend(data)
#     for data in list_data_dict:
#         question = data["question"]
#         answer = data["answer:" if "answer:" in data else "answer"]
#         if 'Jones was born on May 13, 1931' in question:
#             print(question)
#             print(answer)
#             sys.stdin.read(1)

# test_text_only()


# test_string = '''
# [[{"from": "human", "value": "<image>\nWho is the author of this book?\nAnswer the question using a single word or phrase."}, {"from": "gpt", "value": "*"}, {"from": "human", "value": "What is the title of this book?"}, {"from": "gpt", "value": "Rough Guide Map of South Africa, Lesotho"}, {"from": "human", "value": "What is the genre of this book?"}, {"from": "gpt", "value": "Travel"}, {"from": "human", "value": "Is this book related to Travel?"}, {"from": "gpt", "value": "Yes"}, {"from": "human", "value": "Is this book related to History?"}, {"from": "gpt", "value": "No"}]]
# '''

# test_string2 = '''
# [[{"from": "human", "value": "Answer the question at the end by quoting:\n\nJones was born on May 13, 1931 in a rural area of Crete, Indiana, to James Thurman Jones (1887-1951), a World War I veteran, and Lynetta Putnam (1902-1977). Jones was of Irish and Welsh descent; he later claimed partial Cherokee ancestry through his mother, but his maternal second cousin later stated this was likely untrue. Economic difficulties during the Great Depression necessitated that Jones\' family move to the town of Lynn in 1934, where he grew up in a shack without plumbing.\nIn 1951, Jones began attending gatherings of the Communist Party USA in Indianapolis. He became flustered with harassment he received during the McCarthy Hearings, particularly regarding an event he attended with his mother focusing on Paul Robeson, after which she was harassed by the FBI in front of her co-workers for attending. He also became frustrated with ostracism of open communists in the United States, especially during the trial of Julius and Ethel Rosenberg. This frustration, among other things, provoked a seminal moment for Jones in which he asked himself, 'How can I demonstrate my Marxism? The thought was, infiltrate the church.'  Jones was surprised when a Methodist superintendent helped him get a start in the church even though he knew Jones to be a communist and Jones did not meet him through the Communist Party USA. In 1952, he became a student pastor in Sommerset Southside Methodist Church, but claimed he left that church because its leaders barred him from integrating blacks into his congregation. Around this time, Jones witnessed a faith-healing service at a Seventh Day Baptist Church. He observed that it attracted people and their money and concluded that, with financial resources from such healings, he could help accomplish his social goals.  Jones organized a mammoth religious convention to take place on June 11 through June 15, 1956, in a cavernous Indianapolis hall called Cadle Tabernacle. To draw the crowds, Jim needed a religious headliner, and so he arranged to share the pulpit with Rev. William M. Branham, a healing evangelist and religious author who at the time was as highly revered as Oral Roberts. Following the convention, Jones was able to launch his own church, which changed names until it became the Peoples Temple Christian Church Full Gospel. The Peoples Temple was initially made as an inter-racial mission.\n\nHow did he help?\n\nIn 1952, he became a student pastor in Sommerset Southside Methodist Church,\n\nSome context: Modern Sounds in Country and Western Music is a studio album by American R&B singer-songwriter and musician Ray Charles. It was recorded by Charles in February 1962 at Capitol Studios in New York City and at United Recording Studios in Hollywood, then released in April of that year by ABC-Paramount Records. The album departed stylistically from the singer\'s previous rhythm and blues music. It featured country, folk, and Western music standards reworked by Charles in popular song forms of the time, including R&B, pop, and jazz.\nModern Sounds in Country and Western Music was the 18th overall LP Charles had recorded. According to him, the title of the album was conceived by producer Sid Feller and ABC-Paramount\'s executives and management people. The recording sessions for the album took place at three sessions in mid-February 1962. The first two sessions were set on February 5 and 7 at Capitol Studios in New York, New York, at which one half of the album was recorded and produced. The other half was recorded on February 15 of that same year at United Recording Studios in Hollywood, California. Instead of drawing what he should record from memory and his knowledge of country music, Charles asked Feller, his newly appointed A&R (Artists and Repertoire) man, to research top country standards through major country music publishers.  By canvassing premier country publishing companies, such as Acuff-Rose Publishing (which featured the Hank Williams catalog) and Hill & Range Songs (most of which were located in Nashville, Tennessee), Feller amassed around 250 songs on tape for Charles to consider recording for Modern Sounds in Country and Western Music. From New York City, Feller sent the recordings to Charles, who was living in California at the time, for him to choose. According to music essayist Daniel Cooper:  While his selections provided the album's country and western foundation, the musical arrangements represented its contemporary influence. Eager to display his big band ensemble in studio, Charles enlisted premier jazz arrangers Gerald Wilson and Gil Fuller, while Marty Paich, who was active in the West Coast jazz scene, was hired to arrange the lush strings and chorus numbers. Despite enlisting a roster of professional arrangers and musicians, Charles intended to control the artistic direction of the recordings. To indicate specific licks he wanted emphasized for certain songs, Charles would put together voice-and-piano demos and pass them along to the arrangers, informing them of what he wanted to do with specific sounds. According to Feller, at one point during recording, Charles rewrote an entire botched arrangement and dictated the parts to each of the 18 backing musicians.\nAre there any other interesting aspects about this article?\nA: Feller amassed around 250 songs on tape for Charles to consider recording for Modern Sounds in Country and Western Music.\n\nIN: Sean Patrick Hannity was born in New York City, New York, the son of Lillian (Flynn) and Hugh Hannity. Lillian worked as a stenographer and a corrections officer at a county jail, while Hugh was a family-court officer. He is the youngest of four siblings. All of his grandparents immigrated to the United States from Ireland.\n\nHannity hosted his first talk radio show in 1989 at the volunteer college station at UC Santa Barbara, KCSB-FM, while working as a general contractor. The show aired for 40 hours of air time. Regarding his first show, he said, 'I wasn\'t good at it. I was terrible.' Hannity\'s weekly show on KCSB was canceled after less than a year. This was after two shows featuring the book The AIDS Coverup: The Real and Alarming Facts about AIDS by Gene Antonio; among other remarks made during the broadcast, Hannity told a lesbian caller, 'I feel sorry for your child.' The university board that governed the station later reversed its decision due to a campaign conducted on Hannity\'s behalf by the Santa Barbara chapter of the American Civil Liberties Union, which argued that the station had discriminated against Hannity\'s First Amendment rights. When the station refused to give him a public apology and more airtime, Hannity decided against returning to KCSB.  After leaving KCSB, Hannity placed an ad in radio publications presenting himself as 'the most talked about college radio host in America.' Radio station WVNN in Athens, Alabama (part of the Huntsville market), then hired him to be the afternoon talk show host. From Huntsville, he moved to WGST in Atlanta in 1992, filling the slot vacated by Neal Boortz, who had moved to competing station WSB. In September 1996, Fox News co-founder Roger Ailes hired the then relatively unknown Hannity to host a television program under the working title Hannity and LTBD ('liberal to be determined'). Alan Colmes was then hired to co-host and the show debuted as Hannity & Colmes.  Later that year, Hannity left WGST for New York, where WABC had him substitute for their afternoon drive time host during Christmas week. In January 1997, WABC put Hannity on the air full-time, giving him the late night time slot. WABC then moved Hannity to the same drive time slot he had filled temporarily a little more than a year earlier. Hannity was on WABC's afternoon time slot from January 1998 until the end of 2013. Since January 2014, Hannity has hosted the 3-6 p.m. time slot on WOR in New York City.  In their 2007 book Common Ground: How to Stop the Partisan War That Is Destroying America, conservative Cal Thomas and liberal Bob Beckel describe Hannity as a leader of the pack among broadcasting political polarizers, which following James Q. Wilson they define as those who have 'an intense commitment to a candidate, a culture, or an ideology that sets people in one group definitively apart from people in another, rival group.'\n\nwhat year was that\n\nOUT:"}, {"from": "gpt", "value": ""}]]
# '''

# def test_preprocess():
#     tokenizer = transformers.AutoTokenizer.from_pretrained(
#         '/home/jasonlu/models/Mistral-7B-v0.1',
#         cache_dir='',
#         model_max_length=4096,
#         padding_side="right",
#         use_fast=False,
#         legacy=False,
#     )
#     tokenizer.pad_token = tokenizer.unk_token
#     tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)

#     dataset.preprocess_v1(
#         json.loads(test_string, strict=False), tokenizer=tokenizer, n_image_tokens=576, conv_version="vicuna_v1_1")

#     dataset.preprocess_v1(
#         json.loads(test_string2, strict=False), tokenizer=tokenizer, n_image_tokens=576, conv_version="vicuna_v1_1_nosys")

# test_preprocess()

# def test_LazySupervisedDataset():
#     datasets_mixture.register_datasets_mixtures()
#     image_processor = transformers.CLIPImageProcessor.from_pretrained(
#         'openai/clip-vit-large-patch14-336', torch_dtype=torch.float16
#     )

#     tokenizer = transformers.AutoTokenizer.from_pretrained(
#         '/home/jil/models/vicuna-1.5/vicuna-7b-v1.5',
#         cache_dir='',
#         model_max_length=4096,
#         padding_side="right",
#         use_fast=False,
#         legacy=False,
#     )
#     tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)

#     #for dataset_name in ['valley', 'llava_1_5_sft']:
#     for dataset_name in ['valley']:
#         test_dataset = datasets_mixture.DATASETS[dataset_name]

#         multimodal_cfg = {}
#         multimodal_cfg['image_folder'] = test_dataset.image_path
#         multimodal_cfg['is_multimodal'] = True
#         multimodal_cfg['patch_size'] = 14
#         multimodal_cfg['num_shots'] = 0
#         multimodal_cfg["n_extra_patch"] = 0
#         multimodal_cfg["image_aspect_ratio"] = 'square'
#         multimodal_cfg["image_processor"] = image_processor
#         multimodal_cfg["use_im_start_end"] = False

#         conversation_lib.default_conversation = conversation_lib.conv_templates[
#                 "vicuna_v1_1"
#             ]

#         item = dataset.LazySupervisedDataset(data_path=test_dataset.data_path,
#                                             tokenizer=tokenizer,
#                                             multimodal_cfg=multimodal_cfg)

#         print(item.__getitem__(73))

# def test_make_supervised_data_module():
#     datasets_mixture.register_datasets_mixtures()
#     tokenizer = transformers.AutoTokenizer.from_pretrained(
#         '/home/jil/models/vicuna-1.5/vicuna-7b-v1.5',
#         cache_dir='',
#         model_max_length=4096,
#         padding_side="right",
#         use_fast=False,
#         legacy=False,
#     )
#     data_module, extra_info = dataset.make_supervised_data_module(
#         tokenizer=tokenizer,
#         data_args=arguments.DataArguments(
#             datasets_mixture_name='coyo_25m_mmc4core_test',
#             is_multimodal=True,
#             lazy_preprocess=True,
#         ),
#         patch_size=14,
#         n_extra_patch=0,
#     )
#     pp = pprint.PrettyPrinter(indent=2)
#     pp.pprint(data_module)
#     pp.pprint(extra_info)


# def test_DataCollatorForSupervisedDataset():

#     tokenizer = transformers.AutoTokenizer.from_pretrained(
#         '/home/jil/models/vicuna-1.5/vicuna-7b-v1.5',
#         cache_dir='',
#         model_max_length=4096,
#         padding_side="right",
#         use_fast=False,
#         legacy=False,
#     )

#     data_collator = dataset.DataCollatorForSupervisedDataset(
#         tokenizer=tokenizer,
#         concat_prob=0.0
#     )

#     instances = [{
#         'input_ids': torch.ones(1, 5)*3,
#         'labels': torch.ones(1, 5)*1000,
#         'image': torch.ones(4, 3, 4, 4)*10,
#     },{
#         'input_ids': torch.ones(1, 6)*5,
#         'labels': torch.ones(1, 6)*1001,
#         'image': torch.ones(5, 3, 4, 4)*11,
#     },{
#         'input_ids': torch.ones(4, 2)*4,
#         'labels': torch.ones(4, 2)*1002,
#         'image': torch.ones(4, 3, 4, 4)*12,
#     },{
#         'input_ids': torch.ones(1, 3)*6,
#         'labels': torch.ones(1, 3)*1003,
#         'image': torch.ones(6, 3, 4, 4)*13,
#     },
#     ]

#     batch = data_collator(instances)
#     pp = pprint.PrettyPrinter(indent=2)
#     pp.pprint(batch)
#     # Expected result
#     # {
#     #     input_ids=[[5, 5, 5, 5, 5, 5], [3, 3, 3, 3, 3, -1], [6, 6, 6, 4, 4, -1], [4, 4, 4, 4, 4, 4]],
#     #     labels=[[1001, 1001, 1001, 1001, 1001, 1001], [1000, 1000, 1000, 1000, 1000, -1], [1003, 1003, 1003, 1002, 1002, -1], [1002, 1002, 1002, 1002, 1002, 1002]],
#     #     attention_mask=[],
#     #     seqlens_in_batch=[6, 5, 3, 2, 2, 2, 2],
#     #     images = [11 * 5, 10 * 4, 13 * 6, 12, 12, 12, 12],
#     #     position_ids=[[   0,    1,    2,    3,    4,    5],
#             # [   0,    1,    2,    3,    4, -100],
#             # [   0,    1,    2,    0,    1, -100],
#             # [   0,    1,    0,    1,    0,    1]]
#     # }

# test_make_supervised_data_module()
# test_DataCollatorForSupervisedDataset()
# test_LazySupervisedDataset()
