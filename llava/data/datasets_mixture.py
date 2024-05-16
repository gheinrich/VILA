from dataclasses import dataclass, field


@dataclass
class Dataset:
    dataset_name: str
    dataset_type: str = field(default="torch")
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    meta_path: str = field(default=None, metadata={"help": "Path to the meta data for webdataset."})
    image_path: str = field(default=None, metadata={"help": "Path to the training image data."})
    description: str = field(
        default=None,
        metadata={
            "help": "Detailed desciption of where the data is from, how it is labelled, intended use case and the size of the dataset."
        },
    )
    test_script: str = (None,)
    maintainer: str = (None,)


DATASETS = {}

import warnings


def add_dataset(dataset):
    if dataset.dataset_name in DATASETS:
        # make sure the data_name is unique
        warnings.warn(f"{dataset.dataset_name} already existed in DATASETS. Make sure the name is unique.")
    assert "+" not in dataset.dataset_name, "Dataset name cannot include symbol '+'."
    DATASETS.update({dataset.dataset_name: dataset})


def register_datasets_mixtures():
    panda70m_testing = Dataset(
        dataset_name="panda70m_testing",
        dataset_type="panda70m",
        data_path="/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/dataset/panda70m/wds-testing",
        description="",
    )
    add_dataset(panda70m_testing)
    
    panda70m = Dataset(
        dataset_name="panda70m",
        dataset_type="panda70m",
        data_path="/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/dataset/panda70m/wds-training_2m",
        description="",
    )
    add_dataset(panda70m)

    hiertext = Dataset(
        dataset_name="hiertext",
        dataset_type="hiertext",
        data_path="/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/dataset/hiertext",
        description="https://github.com/google-research-datasets/hiertext OCR dataset",
    )
    add_dataset(hiertext)

    textocr = Dataset(
        dataset_name="textocr",
        dataset_type="textocr",
        data_path="/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/dataset/TextOCR",
        description="https://textvqa.org/textocr/ ",
    )
    add_dataset(textocr)

    sam_recap = Dataset(
        dataset_name="sam_recap",
        dataset_type="sam-wds",
        data_path="/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/dataset/sam-reformat",
        description="",
    )
    add_dataset(sam_recap)

    datacomp_webds = Dataset(
        dataset_name="datacomp_webds",
        dataset_type="coyo-wds",
        # data_path='/lustre/fsw/portfolios/llmservice/users/dannyy/dannyy_gpt4/data_filtering/dc1b_filtered',
        # NOTE(ligeng) change to ligeng's path to keep consisty across draco and cs.
        # TODO(ligeng) move to nvr_elm_llm workspace later.
        data_path="/home/ligengz/datasets/dc1b_filtered",
        description="Original data source: https://github.com/mlfoundations/datacomp that contains 1B samples, ranked according to CLIP score and choose the top 18M. Short Image - Text pairs.",
    )
    add_dataset(datacomp_webds)

    coyo_webds_refilerted = Dataset(
        dataset_name="coyo_webds_refilerted",
        dataset_type="coyo-wds",
        # data_path='/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/datasets/captioning/coyo-700m_full_webdata_fullmeta/stage2_filtered_v2',
        # NOTE(ligeng) change to ligeng's path to keep consisty across draco and cs.
        # TODO(ligeng) move to nvr_elm_llm workspace later.
        data_path="/home/ligengz/datasets/coyo-refilter",
        description="Original data source: https://github.com/kakaobrain/coyo-dataset that contains 700M samples, ranked according to CLIP score and choose the top 20M. Short Image - Text pairs.",
    )
    add_dataset(coyo_webds_refilerted)

    coyo_webds_vila_recaption = Dataset(
        dataset_name="coyo_25m_wds_recap",
        dataset_type="coyo-wds-recap",
        # data_path='/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/datasets/captioning/coyo-25m-vila',
        # NOTE(ligeng) change to ligeng's path to keep consisty across draco and cs.
        # TODO(ligeng) move to nvr_elm_llm workspace later.
        data_path="/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/datasets/captioning/coyo-25m-vila",
        description="See coyo. Relabel coyo w/ VILA captioner, long Image - Text pair.",
    )
    add_dataset(coyo_webds_vila_recaption)
    
    coyo_webds_vila_recaption_5_subset = Dataset(
        dataset_name="coyo_25m_wds_recap_5_subset",
        dataset_type="coyo-wds-recap",
        data_path="/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/datasets/captioning/coyo-25m-vila",
        meta_path="/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/datasets/captioning/coyo-25m-vila/wids-meta-5-subset.json",
        description="5% subset of coyo_webds_vila_recaption.",
    )
    add_dataset(coyo_webds_vila_recaption_5_subset)
    
    coyo_webds_vila_recaption_10_subset = Dataset(
        dataset_name="coyo_25m_wds_recap_10_subset",
        dataset_type="coyo-wds-recap",
        data_path="/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/datasets/captioning/coyo-25m-vila",
        meta_path="/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/datasets/captioning/coyo-25m-vila/wids-meta-10-subset.json",
        description="5% subset of coyo_webds_vila_recaption.",
    )
    add_dataset(coyo_webds_vila_recaption_10_subset)

    # data_path='/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/datasets/captioning/coyo-700m_full_webdata',
    # data_path='/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/datasets/captioning/coyo-25m-vila',
    coyo_25m_wds = Dataset(
        dataset_name="coyo_25m_wds",
        dataset_type="coyo-wds",
        data_path="/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/dataset/coyo-25m-vila",
        description="See coyo. Convert coyo to webds format.",
    )
    add_dataset(coyo_25m_wds)
    
    coyo_25m_wds_5_subset = Dataset(
        dataset_name="coyo_25m_wds_5_subset",
        dataset_type="coyo-wds",
        data_path="/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/dataset/coyo-25m-vila",
        meta_path="/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/dataset/coyo-25m-vila/wids-meta-5-subset.json",
        description="5% subset of coyo_25m_wds.",
    )
    add_dataset(coyo_25m_wds_5_subset)
    
    coyo_25m_wds_10_subset = Dataset(
        dataset_name="coyo_25m_wds_10_subset",
        dataset_type="coyo-wds",
        data_path="/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/dataset/coyo-25m-vila",
        meta_path="/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/dataset/coyo-25m-vila/wids-meta-10-subset.json",
        description="10% subset of coyo_25m_wds.",
    )
    add_dataset(coyo_25m_wds_10_subset)
    
    coyo_webds_full = Dataset(
        dataset_name="coyowebds_full",
        dataset_type="coyo-wds",
        data_path="/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/datasets/captioning/coyo-700m_full_webdata",
        description="Full coyo700M. Data source: https://github.com/kakaobrain/coyo-dataset, short Image - Text pair.",
    )
    add_dataset(coyo_webds_full)
    ############################################################################################

    coyo_25m = Dataset(
        dataset_name="coyo_25m",
        dataset_type="coyo",
        data_path="/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/dataset/vila-10-dataset/coyo-700m/pkl02-split",
        description="Original data source: https://github.com/kakaobrain/coyo-dataset that contains 700M samples, ranked according to CLIP score (per shard) and choose the top 25M. Short Image - Text pairs.",
    )
    add_dataset(coyo_25m)
    coyo_25m_test = Dataset(
        dataset_name="coyo_25m_test",
        dataset_type="coyo",
        data_path="/home/jasonlu/vlm_datasets/debug/coyo-700m/pkl02-split",
        description="See coyo. A subset of coyo (16 shards) that could be used for test purposes.",
    )
    add_dataset(coyo_25m_test)

    # TODO: switch mmc4 to wds impl as well.
    mmc4core = Dataset(
        dataset_name="mmc4core",
        dataset_type="mmc4",
        # data_path='/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/dataset/vila-10-dataset/mmc4-core/pkl-core',
        data_path="/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/dataset/vila-10-dataset/mmc4-core/pkl-core",
        description="Original data source: https://github.com/allenai/mmc4 mmc4-core that contains 29.9M images, interleaved Image - Text data.",
    )
    add_dataset(mmc4core)

    mmc4core_10_subset = Dataset(
        dataset_name="mmc4core_10_subset",
        dataset_type="mmc4",
        data_path="/home/yunhaof/workspace/datasets/subsets/mmc4core_subset",
        description="10% subset of mmc4core.",
    )
    add_dataset(mmc4core_10_subset)
    
    mmc4core_test = Dataset(
        dataset_name="mmc4core_test",
        dataset_type="mmc4",
        data_path="/home/jasonlu/vlm_datasets/debug/mmc4-core/pkl-core",
        description="See mmc4core. A subset of mmc4core (16 shards) that could be used for test purposes.",
    )
    add_dataset(mmc4core_test)

    ccs_recap_wds = Dataset(
        dataset_name="ccs_recap_wds",
        dataset_type="ccs-wds",
        data_path="/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/dataset/ccs_recaptioned",
        description="TODO dannyy",
    )
    add_dataset(ccs_recap_wds)

    ccs_recaptioned = Dataset(
        dataset_name="ccs_recaptioned",
        dataset_type="wds",
        data_path="/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/dataset/vila-10-dataset/ccs_recaptioned",
        description="TODO dannyy",
    )
    add_dataset(ccs_recaptioned)

    ccs_recaptioned_test = Dataset(
        dataset_name="ccs_recaptioned_test",
        dataset_type="wds",
        data_path="/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/dataset/vila-10-dataset/ccs_recaptioned_test",
        description="See ccs_recaptioned, A subset of ccs_recaptioned (16 shards) that could be used for test purposes.",
    )
    add_dataset(ccs_recaptioned_test)

    vflan = Dataset(
        dataset_name="vflan",
        dataset_type="vflan",
        data_path="/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/dataset/vila-10-dataset/vlm-flan-clean-text1m-nosqa",
    )
    add_dataset(vflan)

    llava_1_5_mm_align = Dataset(
        dataset_name="llava_1_5_mm_align",
        dataset_type="torch",
        data_path="/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/dataset/LLaVA-CC3M-Pretrain-595K/chat.json",
        image_path="/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/dataset/LLaVA-CC3M-Pretrain-595K/images",
    )
    add_dataset(llava_1_5_mm_align)
    
    llava_1_5_pretrain = Dataset(
        dataset_name="llava_1_5_pretrain",
        dataset_type="torch",
        data_path="/home/yunhaof/workspace/datasets/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json",
        image_path="/home/yunhaof/workspace/datasets/LLaVA-Pretrain/images",
    )
    add_dataset(llava_1_5_pretrain)
    
    llava_1_5_sft = Dataset(
        dataset_name="llava_1_5_sft",
        dataset_type="torch",
        data_path="./playground/data/llava_v1_5_mix665k.json",
        image_path="./playground/data",
    )
    add_dataset(llava_1_5_sft)

    sharegpt4v_sft = Dataset(
        dataset_name="sharegpt4v_sft",
        dataset_type="torch",
        data_path="/home/jasonlu/vlm_datasets/ShareGPT4V/jason-filter-sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json",
        image_path="/home/jasonlu/vlm_datasets/ShareGPT4V/data",
        description="Original data source: https://sharegpt4v.github.io/ 655K llava_1_5_sft data relablled w/ ShareGPT4V captioner.",
    )
    add_dataset(sharegpt4v_sft)

    sharegpt4v_gpt4_100k = Dataset(
        dataset_name="sharegpt4v_gpt4_100k",
        dataset_type="torch",
        data_path="/home/jasonlu/vlm_datasets/ShareGPT4V/jason-filter-sharegpt4v_instruct_gpt4-vision_cap100k.json",
        image_path="/home/jasonlu/vlm_datasets/ShareGPT4V/data",
        description="Original data source: https://sharegpt4v.github.io/ ~100K long Image - Text pair generated by GPT4V.",
    )
    add_dataset(sharegpt4v_gpt4_100k)
    
    allava_caption_vflan = Dataset(
        dataset_name="allava_caption_vflan",
        dataset_type="torch",
        data_path="/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/dataset/ALLaVA-4V/ALLaVA-Caption-VFLAN-4V.json",
        image_path="/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/dataset/ALLaVA-4V/",
    )
    add_dataset(allava_caption_vflan)

    allava_instruct_vflan = Dataset(
        dataset_name="allava_instruct_vflan",
        dataset_type="torch",
        data_path="/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/dataset/ALLaVA-4V/ALLaVA-Instruct-VFLAN-4V.json",
        image_path="/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/dataset/ALLaVA-4V/",
    )
    add_dataset(allava_instruct_vflan)

    chartqa = Dataset(
        dataset_name="chartqa",
        dataset_type="torch",
        data_path="/home/yunhaof/workspace/datasets/ChartQA/processed/train_merged_28k.json",
        image_path="/home/yunhaof/workspace/datasets/ChartQA/train/png",
    )
    add_dataset(chartqa)

    llavar = Dataset(
        dataset_name="llavar",
        dataset_type="torch",
        data_path="/home/yunhaof/workspace/datasets/LLaVAR-Instruct-16K/llavar_16k_instruction_finetune.json",
        image_path="/home/yunhaof/workspace/datasets/LLaVAR-Instruct-16K/images",
    )
    add_dataset(llavar)

    dvqa = Dataset(
        dataset_name="dvqa",
        dataset_type="torch",
        data_path="/home/yunhaof/workspace/datasets/DVQA/processed/DVQA_train_qa_2325k.json",
        image_path="/home/yunhaof/workspace/datasets/DVQA/images",
    )
    add_dataset(dvqa)
    
    dvqa_subset = Dataset(
        dataset_name="dvqa_subset",
        dataset_type="torch",
        data_path="/home/yunhaof/workspace/datasets/DVQA/processed/DVQA_train_qa_subset100K.json",
        image_path="/home/yunhaof/workspace/datasets/DVQA/images",
    )
    add_dataset(dvqa_subset)

    ai2d = Dataset(
        dataset_name="ai2d",
        dataset_type="torch",
        data_path="/home/yunhaof/workspace/datasets/AI2D/processed/train_12k.json",
        image_path="/home/yunhaof/workspace/datasets/AI2D/ai2d/images",
    )
    add_dataset(ai2d)

    # synthdog_en = Dataset(
    #     dataset_name="synthdog_en",
    #     dataset_type="torch",
    #     data_path="/home/yunhaof/workspace/datasets/synthdog-en/synthdog_en_66_5k_with_question.json",
    #     image_path="/home/yunhaof/workspace/datasets/synthdog-en/images",
    # )
    # add_dataset(synthdog_en)

    visual7w = Dataset(
        dataset_name="visual7w",
        dataset_type="torch",
        data_path="/home/yunhaof/workspace/datasets/visual7w/processed/v7w_pointing_train.json",
        image_path="/home/yunhaof/workspace/datasets/visual7w/images",
    )
    add_dataset(visual7w)

    shikra = Dataset(
        dataset_name="shikra",
        dataset_type="torch",
        data_path="/home/yunhaof/workspace/datasets/shikra/shikra.json",
        image_path="/home/jasonlu/vlm_datasets/flickr30k-images",
    )
    add_dataset(shikra)

    scienceqa = Dataset(
        dataset_name="scienceqa",
        dataset_type="torch",
        data_path="/home/yunhaof/workspace/datasets/evaluation/scienceqa/scienceqa_train_12k.json",
        image_path="/home/yunhaof/workspace/datasets/evaluation/scienceqa/images",
    )
    add_dataset(scienceqa)

    grit_mixture = Dataset(
        dataset_name="grit_mixture",
        dataset_type="torch",
        data_path="/home/yunhaof/workspace/datasets/GRIT/processed-grit-2m/filtered_grit_merged_885k.json",
        image_path="/home/yunhaof/workspace/datasets/GRIT/processed-grit-2m/webdataset_untar",
    )
    add_dataset(grit_mixture)

    grit_grounding = Dataset(
        dataset_name="grit_grounding",
        dataset_type="torch",
        data_path="/home/yunhaof/workspace/datasets/GRIT/processed-grit-2m/single_grounding_qa_1177k.json",
        image_path="/home/yunhaof/workspace/datasets/GRIT/processed-grit-2m/webdataset_untar",
    )
    add_dataset(grit_grounding)

    sharegpt4v_pretrain = Dataset(
        dataset_name="sharegpt4v_pretrain",
        dataset_type="torch",
        data_path="/home/jasonlu/vlm_datasets/ShareGPT4V/jason-filter-share-captioner_coco_lcs_sam_1246k_1107.json",
        image_path="/home/jasonlu/vlm_datasets/ShareGPT4V/data",
        description="Original data source: https://sharegpt4v.github.io/ ~1M long Image - Text pair generated by ShareGPT4V captioner.",
    )
    add_dataset(sharegpt4v_pretrain)

    mmmu_validation = Dataset(
        dataset_name="mmmu_validation",
        dataset_type="evaluation",
        data_path="./playground/data/eval/MMMU",
        description="MMMU validation set.",
    )
    add_dataset(mmmu_validation)

    valley = Dataset(
        dataset_name="valley",
        dataset_type="torch",
        data_path="/home/jasonlu/video_datasets/Webvid/chat.json",
        image_path="/home/jasonlu/video_datasets/Webvid/data/videos/",
        description="Original data source: https://huggingface.co/datasets/luoruipu1/Valley-webvid2M-Pretrain-703K, 703K data collected and filtered from Webvid-2M.",
    )
    add_dataset(valley)
    
    video_chatgpt = Dataset(
        dataset_name="video_chatgpt",
        dataset_type="torch",
        data_path="/home/jasonlu/video_datasets/Video_ChatGPT/VideoInstruct-100K/filtered_VideoInstruct100K.json",
        image_path="/home/jasonlu/video_datasets/Video_ChatGPT/activitynet_videos/",
        description="Original data source: https://github.com/mbzuai-oryx/Video-ChatGPT/blob/main/README.md#video-instruction-dataset-open_file_folder, 100K human-assisted and semi-automatic video instruction dataset.",
    )
    add_dataset(video_chatgpt)
    
    valley_test = Dataset(
        dataset_name="valley_test",
        dataset_type="torch",
        data_path="/home/jasonlu/video_datasets/Webvid/chat_test.json",
        image_path="/home/jasonlu/video_datasets/Webvid/data/videos/",
        description="See valley, A subset of valley (16 shards) that could be used for test purposes.",
    )
    add_dataset(valley_test)
    
    jukinmedia = Dataset(
        dataset_name="jukinmedia",
        dataset_type="torch",
        # data_path="/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets_v2/jukinmedia/jukin-100k-filtered-bin.json",
        # image_path="/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets_v2/jukinmedia/videos_decompress_v2",
        data_path="/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/dataset/jukinmedia/jukin-100k-filtered-bin.json",
        image_path="/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/dataset/jukinmedia/videos_decompress_v2",
        description="A high quailty video caption dataset with 71018 detailed captions. See READMD.md file for the details (e.g. prompt template) of the dataset.",
    )
    add_dataset(jukinmedia)
    
    youcook2 = Dataset(
        dataset_name="youcook2",
        dataset_type="torch",
        # /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/dataset/video_datasets_v2
        # data_path="/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets_v2/youcook2/youcookii_clipped-v2.json",
        # image_path="/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets_v2/youcook2/video_data_clipped",
        data_path="/home/jasonlu/video_datasets/jason_filtered_youcook2.json",
        image_path="/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/dataset/video_datasets_v2/youcook2/video_data_clipped",
        description="YouCook2 (http://youcook2.eecs.umich.edu/): A large-scale video dataset with 11680 short but precise human written captions.",
    )
    add_dataset(youcook2)
    
    vatex = Dataset(
        dataset_name="vatex",
        dataset_type="torch",
        # /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/dataset/video_datasets_v2
        # data_path="/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets_v2/vatex/vatex_training_processed_filtered-v2.json",
        # image_path="/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets_v2/vatex/videos_clipped",
        data_path="/home/jasonlu/video_datasets/jason_filtered_vatex.json",
        image_path="/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/dataset/video_datasets_v2/vatex/videos_clipped",
        description="VATEX dataset (https://eric-xw.github.io/vatex-website/about.html), 22703 video clips, 227030 precise short captions (human annotated). Note: all clips are 10s.",
    )
    add_dataset(vatex)
    activitynet_qa = Dataset(
        dataset_name="activitynet_qa",
        dataset_type="torch",
        data_path="/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets_v2/activitynet-qa/train-processed-filtered-v2.json",
        image_path="/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets/Video_ChatGPT/activitynet_videos",
        description="28250 human-annotated QA pairs on 2825 videos derived from the popular ActivityNet dataset.",
    )
    add_dataset(activitynet_qa)
    ivqa = Dataset(
        dataset_name="ivqa",
        dataset_type="torch",
        data_path="/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets_v2/ivqa/train-processed-filtered.json",
        image_path="/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets_v2/ivqa/video_data_clipped",
        description="iVQA dataset, 5378 videos with 5378 QA pairs. The 5378 QA pairs are from various domains.",
    )
    add_dataset(ivqa)
    nextqa = Dataset(
        dataset_name="nextqa",
        dataset_type="torch",
        data_path="/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets_v2/nextqa/train-processed.json",
        image_path="/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets_v2/nextqa/NExTVideo",
        description="NextQA dataset(https://github.com/doc-doc/NExT-QA/tree/main), 34132 human annotated questions from various domains.",
    )
    add_dataset(nextqa)
    msrvttqa = Dataset(
        dataset_name="msrvttqa",
        dataset_type="torch",
        data_path="/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets_v2/msr_vtt/train-processed-qa-v2.json",
        image_path="/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets_v2/msr_vtt/train_val_videos/TrainValVideo",
        description="6321 videos with 6321 rewritten QA-pairs based on the rewritten captions. (The typos in captions have been fixed by GPT-3.5-turbo)",
    )
    add_dataset(msrvttqa)
    
    shot2story_shotonly = Dataset(
        dataset_name="shot2story_shotonly",
        dataset_type="torch",
        # data_path="/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets_v2/shot2story/train-shortclip-processed-bin.json",
        # image_path="/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/Shot2Story/data/videos_extracted",
        data_path="/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/dataset/shot2story/train-shortclip-processed-bin.json",
        image_path="/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/dataset/shot2story/Shot2Story/data/videos_extracted",
        description="48K high quality video clips with 48K short or long high-qualiy captions.",
    )
    add_dataset(shot2story_shotonly)
    sharegpt_video = Dataset(
        dataset_name="sharegpt_video",
        dataset_type="torch",
        # /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/dataset/video_datasets_v2
        # data_path="/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets_v2/sharegpt_video/video_caption_pretrain.json",
        # image_path="/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets_v2/sharegpt_video/videos",
        data_path="/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/dataset/video_datasets_v2/sharegpt_video/video_caption_pretrain.json",
        image_path="/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/dataset/video_datasets_v2/sharegpt_video/videos",
        description="900K high quailty detailed video caption written by GPT-4V",
    )
    add_dataset(sharegpt_video)

    sharegpt_video_qa = Dataset(
        dataset_name="sharegpt_video_qa",
        dataset_type="torch",
        # /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/dataset/video_datasets_v2
        # data_path="/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets_v2/sharegpt_video/video_caption_pretrain.json",
        # image_path="/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets_v2/sharegpt_video/videos",
        data_path="/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/dataset/video_datasets_v2/sharegpt_video/chatgpt_qa_900k.json",
        image_path="/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/dataset/video_datasets_v2/sharegpt_video/videos",
        description="900K high quailty detailed video caption written by GPT-4V",
    )
    add_dataset(sharegpt_video_qa)

    # Video Pretraining Datasets added by Fuzhao
    internvid_test = Dataset(
        dataset_name="internvid_test",
        dataset_type="video-wds",
        # data_path="/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets_v2/internvid/video_data_tar/InternVid-8K-flt",
        data_path="/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/dataset/video_datasets_v2/internvid/video_data_tar/InternVid-8K-flt",
        description="A tiny debug set of internvid with only 8K samples.",
    )
    add_dataset(internvid_test)
    
    internvid_1300K = Dataset(
        dataset_name="internvid_1300K",
        dataset_type="video-wds",
        # data_path="/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets_v2/internvid/video_data_tar/InternVid-1300K-flt",
        data_path="/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/dataset/video_datasets_v2/internvid/video_data_tar/InternVid-1300K-flt",
        description="1M (not 1300K after cleaning) video-caption pairs from InternVid. We select the top-relevant 1M samples from the Intern-Vid-10M set.",
    )
    add_dataset(internvid_1300K)
    
    internvid_10M = Dataset(
        dataset_name="internvid_10M",
        dataset_type="video-wds",
        data_path="/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets_v2/internvid/video_data_tar/InternVid-10M-flt",
        # data_path="/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/dataset/video_datasets_v2/internvid/video_data_tar/InternVid-10M-flt",
        description="10M (actually 8M) video-caption pairs from InternVid 10M dataset.",
    )
    add_dataset(internvid_10M)
    
    # TODO(ligeng): syncing to draco
    ego4d_1M = Dataset(
        dataset_name="ego4d_1M",
        dataset_type="video-wds",
        data_path="/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets_v3/ego4d/ego4d_clips_tar/ego4d_1m",
        description="A subset of Ego4D dataset including 1M video-caption pairs. We re-generate the captions by removing the speical characters.",
    )
    add_dataset(ego4d_1M)

    lvis_instruct = Dataset(
        dataset_name="lvis_instruct",
        dataset_type="torch",
        data_path="/home/yunhaof/workspace/datasets/LVIS-Instruct4V/lvis_instruct4v_220k.json",
        image_path="/home/yunhaof/workspace/datasets"
    )
    add_dataset(lvis_instruct)
    
    arxivqa = Dataset(
        dataset_name="arxivqa",
        dataset_type="torch",
        data_path="/home/yunhaof/workspace/datasets/ArxivQA/arxivqa_100k.json",
        image_path="/home/yunhaof/workspace/datasets/ArxivQA",
    )
    add_dataset(arxivqa)

    llava_instruct = Dataset(
        dataset_name="llava_instruct",
        dataset_type="torch",
        data_path="/home/jasonlu/workspace/InternVL/internvl_chat/playground/llava_instruct_150k_zh.jsonl",
        image_path="/home/jasonlu/workspace/InternVL/internvl_chat/playground/data/coco",
        description="",
    )
    add_dataset(llava_instruct)



    dvqa_train_200k = Dataset(
        dataset_name="dvqa_train_200k",
        dataset_type="torch",
        data_path="/home/jasonlu/workspace/InternVL/internvl_chat/playground/dvqa_train_200k.jsonl",
        image_path="/home/jasonlu/workspace/InternVL/internvl_chat/playground/data/dvqa",
        description="",
    )
    add_dataset(dvqa_train_200k)


    chartqa_train_18k = Dataset(
        dataset_name="chartqa_train_18k",
        dataset_type="torch",
        data_path="/home/jasonlu/workspace/InternVL/internvl_chat/playground/chartqa_train_18k.jsonl",
        image_path="/home/jasonlu/workspace/InternVL/internvl_chat/playground/data/chartqa",
        description="",
    )
    add_dataset(chartqa_train_18k)

    ai2d_train_12k = Dataset(
        dataset_name="ai2d_train_12k",
        dataset_type="torch",
        data_path="/home/jasonlu/workspace/InternVL/internvl_chat/playground/ai2d_train_12k.jsonl",
        image_path="/home/jasonlu/workspace/InternVL/internvl_chat/playground/data/ai2d",
        description="",
    )
    add_dataset(ai2d_train_12k)

    docvqa_train_10k = Dataset(
        dataset_name="docvqa_train_10k",
        dataset_type="torch",
        data_path="/home/jasonlu/workspace/InternVL/internvl_chat/playground/docvqa_train_10k.jsonl",
        image_path="/home/jasonlu/workspace/InternVL/internvl_chat/playground/data/docvqa",
        description="",
    )
    add_dataset(docvqa_train_10k)

    geoqa = Dataset(
        dataset_name="geoqa",
        dataset_type="torch",
        data_path="/home/jasonlu/workspace/InternVL/internvl_chat/playground/geoqa+.jsonl",
        image_path="/home/jasonlu/workspace/InternVL/internvl_chat/playground/data/geoqa+",
        description="",
    )
    add_dataset(geoqa)

    synthdog_en = Dataset(
        dataset_name="synthdog_en",
        dataset_type="torch",
        data_path="/home/jasonlu/workspace/InternVL/internvl_chat/playground/synthdog_en.jsonl",
        image_path="/home/jasonlu/workspace/InternVL/internvl_chat/playground/data/synthdog-en",
        description="",
    )
    add_dataset(synthdog_en)

    idefics2_sft = Dataset(
        dataset_name="idefics2_sft",
        dataset_type="torch",
        data_path="/home/jasonlu/workspace/idefics2-sft/new-vflan/idefics2_sft_train.jsonl",
        image_path="/home/jasonlu/workspace/idefics2-sft/new-vflan",
        description="",
    )
    add_dataset(idefics2_sft)

    test = Dataset(
        dataset_name="test",
        dataset_type="torch",
        data_path="/home/jasonlu/workspace/InternVL/internvl_chat/playground/test.jsonl",
        image_path="/home/jasonlu/workspace/InternVL/internvl_chat/playground/data",
        description="",
    )
    add_dataset(test)

    mmc_instruction = Dataset(
        dataset_name="mmc_instruction",
        dataset_type="torch",
        data_path="/home/yunhaof/workspace/datasets/MMC-Instruction/processed/mmc_instruction_410k.json",
        image_path="/home/yunhaof/workspace/datasets/MMC-Instruction",
    )
    add_dataset(mmc_instruction)
    lrv_instruction = Dataset(
        dataset_name="lrv_instruction",
        dataset_type="torch",
        data_path="/home/yunhaof/workspace/datasets/LRV-Instruction/processed/lrv_instruction_321k.json",
        image_path="/home/jasonlu/vlm_datasets/ShareGPT4V/data/vg",
    )
    add_dataset(lrv_instruction)
    sherlock = Dataset(
        dataset_name="sherlock",
        dataset_type="torch",
        data_path="/home/yunhaof/workspace/datasets/sherlock/processed/sherlock_317k.json",
        image_path="/home/yunhaof/workspace/datasets/sherlock/images",
    )
    add_dataset(sherlock)
    math = Dataset(
        dataset_name="math",
        dataset_type="vflan",
        data_path="/home/yunhaof/workspace/datasets/math",
    )
    add_dataset(math)

    geo_qa = Dataset(
        dataset_name="geo_qa",
        dataset_type="torch",
        data_path="/home/yunhaof/workspace/datasets/Geo170K/qa_tuning.json",
        image_path="/home/yunhaof/workspace/datasets/Geo170K/images",
    )
    add_dataset(geo_qa)

    wit_subset = Dataset(
        dataset_name="wit_subset",
        dataset_type="torch",
        data_path="/home/yunhaof/workspace/datasets/WIT/wit_1_8m/wit_processed_538k.json",
        image_path="/home/yunhaof/workspace/datasets/WIT/wit_1_8m/images"
    )
    add_dataset(wit_subset)

    dummy = Dataset(
        dataset_name="dummy",
        dataset_type="dummy",
        data_path="dummy",
        image_path="dummy",
    )
    add_dataset(dummy)
    
    nv_sft = Dataset(
        dataset_name="nv_sft",
        dataset_type="torch",
        data_path="/home/jasonlu/vlm_datasets/nv_sft/project_539_torch.json",
        image_path="/home/jasonlu/vlm_datasets/nv_sft"
    )
    add_dataset(nv_sft)

    # ========================================================
    # datasets for osmo storage
    # ========================================================
    osmo_shot2story_shotonly = Dataset(
        dataset_name="osmo_shot2story_shotonly",
        dataset_type="torch",
        data_path="/mnt/amlfs-01/home/fuzhaox/video_datasets_v2/shot2story_shotonly_v2/train-shortclip-processed-bin.json",
        image_path="/mnt/amlfs-01/home/fuzhaox/video_datasets_v2/shot2story_shotonly_v2/videos_extracted",
        description="48K high quality video clips with 48K short or long high-qualiy captions.",
    )
    add_dataset(osmo_shot2story_shotonly)

    osmo_ccs_recaptioned = Dataset(
        dataset_name="osmo_ccs_recaptioned",
        dataset_type="wds",
        data_path="/mnt/amlfs-01/home/fuzhaox/image_datasets/ccs_recaptioned",
        description="TODO dannyy",
    )
    add_dataset(osmo_ccs_recaptioned)

    osmo_internvid_1300K = Dataset(
        dataset_name="osmo_internvid_1300K",
        dataset_type="video-wds",
        # data_path="/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets_v2/internvid/video_data_tar/InternVid-1300K-flt",
        data_path="/mnt/amlfs-01/home/fuzhaox/video_datasets_v2/internvid/InternVid-1300K-flt",
        # meta_path="/mnt/amlfs-01/home/fuzhaox/video_datasets_v2/internvid/InternVid-1300K-flt/wids-meta.json",
        description="1M (not 1300K after cleaning) video-caption pairs from InternVid. We select the top-relevant 1M samples from the Intern-Vid-10M set.",
    )
    add_dataset(osmo_internvid_1300K)

    osmo_internvid_10M = Dataset(
        dataset_name="osmo_internvid_10M",
        dataset_type="video-wds",
        # data_path="/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets_v2/internvid/video_data_tar/InternVid-1300K-flt",
        data_path="/mnt/amlfs-01/home/fuzhaox/video_datasets_v2/internvid/InternVid-10M-flt",
        # meta_path="/mnt/amlfs-01/home/fuzhaox/video_datasets_v2/internvid/InternVid-1300K-flt/wids-meta.json",
        description="1M (not 1300K after cleaning) video-caption pairs from InternVid. We select the top-relevant 1M samples from the Intern-Vid-10M set.",
    )
    add_dataset(osmo_internvid_10M)

    osmo_coyo_25m = Dataset(
        dataset_name="osmo_coyo_25m",
        dataset_type="coyo",
        data_path="/mnt/amlfs-01/home/fuzhaox/image_datasets/coyo_25m/pkl02-split",
        description="Original data source: https://github.com/kakaobrain/coyo-dataset that contains 700M samples, ranked according to CLIP score (per shard) and choose the top 25M. Short Image - Text pairs.",
    )
    add_dataset(osmo_coyo_25m)

    # TODO: switch mmc4 to wds impl as well.
    osmo_mmc4core = Dataset(
        dataset_name="osmo_mmc4core",
        dataset_type="mmc4",
        # data_path='/home/jasonlu/datasets/mmc4-core/pkl-core',
        data_path="/mnt/amlfs-01/home/fuzhaox/image_datasets/mmc4core/pkl-core",
        description="Original data source: https://github.com/allenai/mmc4 mmc4-core that contains 29.9M images, interleaved Image - Text data.",
    )
    add_dataset(osmo_mmc4core)

    osmo_sharegpt4v_pretrain = Dataset(
        dataset_name="osmo_sharegpt4v_pretrain",
        dataset_type="torch",
        data_path="/mnt/amlfs-01/home/fuzhaox/image_datasets/ShareGPT4V/jason-filter-share-captioner_coco_lcs_sam_1246k_1107.json",
        image_path="/mnt/amlfs-01/home/fuzhaox/image_datasets/ShareGPT4V/data",
        description="Original data source: https://sharegpt4v.github.io/ ~1M long Image - Text pair generated by ShareGPT4V captioner.",
    )
    add_dataset(osmo_sharegpt4v_pretrain)

    osmo_jukinmedia = Dataset(
        dataset_name="osmo_jukinmedia",
        dataset_type="torch",
        # data_path="/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets_v2/jukinmedia/jukin-100k-filtered-bin.json",
        # image_path="/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets_v2/jukinmedia/videos_decompress_v2",
        data_path="/mnt/amlfs-01/home/fuzhaox/video_datasets_v2/jukinmedia/jukinmedia/jukin-100k-filtered-bin.json",
        image_path="/mnt/amlfs-01/home/fuzhaox/video_datasets_v2/jukinmedia/jukinmedia/videos_decompress_v2",
        description="A high quailty video caption dataset with 71018 detailed captions. See READMD.md file for the details (e.g. prompt template) of the dataset.",
    )
    add_dataset(osmo_jukinmedia)

    osmo_panda70m = Dataset(
        dataset_name="osmo_panda70m",
        dataset_type="panda70m",
        data_path="/mnt/amlfs-01/home/fuzhaox/video_datasets_v2/panda2m/wds-training_2m",
        description="",
    )
    add_dataset(osmo_panda70m)

    osmo_sharegpt_video = Dataset(
        dataset_name="osmo_sharegpt_video",
        dataset_type="torch",
        # /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/dataset/video_datasets_v2
        # data_path="/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets_v2/sharegpt_video/video_caption_pretrain.json",
        # image_path="/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets_v2/sharegpt_video/videos",
        data_path="/mnt/amlfs-01/home/fuzhaox/video_datasets_v2/sharegpt_video/sharegpt_video/video_caption_pretrain.json",
        image_path="/mnt/amlfs-01/home/fuzhaox/video_datasets_v2/sharegpt_video/sharegpt_video/videos",
        description="900K high quailty detailed video caption written by GPT-4V",
    )
    add_dataset(osmo_sharegpt_video)

    osmo_sharegpt_video_qa = Dataset(
        dataset_name="osmo_sharegpt_video_qa",
        dataset_type="torch",
        # /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/dataset/video_datasets_v2
        # data_path="/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets_v2/sharegpt_video/video_caption_pretrain.json",
        # image_path="/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets_v2/sharegpt_video/videos",
        data_path="/mnt/amlfs-01/home/fuzhaox/video_datasets_v2/sharegpt_video/sharegpt_video/chatgpt_qa_900k.json",
        image_path="/mnt/amlfs-01/home/fuzhaox/video_datasets_v2/sharegpt_video/sharegpt_video/videos",
        description="900K high quailty detailed video caption written by GPT-4V",
    )
    add_dataset(osmo_sharegpt_video_qa)

    osmo_video_chatgpt = Dataset(
        dataset_name="osmo_video_chatgpt",
        dataset_type="torch",
        data_path="/mnt/amlfs-01/home/fuzhaox/video_datasets_v2/video_chatgpt/video_chatgpt_caption/filtered_VideoInstruct100K.json",
        image_path="/mnt/amlfs-01/home/fuzhaox/video_datasets_v2/video_chatgpt/video_chatgpt_videos/activitynet_videos/",
        description="Original data source: https://github.com/mbzuai-oryx/Video-ChatGPT/blob/main/README.md#video-instruction-dataset-open_file_folder, 100K human-assisted and semi-automatic video instruction dataset.",
    )
    add_dataset(osmo_video_chatgpt)

    osmo_youcook2 = Dataset(
        dataset_name="osmo_youcook2",
        dataset_type="torch",
        # /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/dataset/video_datasets_v2
        # data_path="/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets_v2/youcook2/youcookii_clipped-v2.json",
        # image_path="/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets_v2/youcook2/video_data_clipped",
        data_path="/mnt/amlfs-01/home/fuzhaox/video_datasets_v2/youcook2/jason_filtered_youcook2.json",
        image_path="/mnt/amlfs-01/home/fuzhaox/video_datasets_v2/youcook2/video_data_clipped",
        description="YouCook2 (http://youcook2.eecs.umich.edu/): A large-scale video dataset with 11680 short but precise human written captions.",
    )
    add_dataset(osmo_youcook2)
    
    osmo_vatex = Dataset(
        dataset_name="osmo_vatex",
        dataset_type="torch",
        # /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/dataset/video_datasets_v2
        # data_path="/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets_v2/vatex/vatex_training_processed_filtered-v2.json",
        # image_path="/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets_v2/vatex/videos_clipped",
        data_path="/mnt/amlfs-01/home/fuzhaox/video_datasets_v2/vatex/jason_filtered_vatex.json",
        image_path="/mnt/amlfs-01/home/fuzhaox/video_datasets_v2/vatex/videos_clipped",
        description="VATEX dataset (https://eric-xw.github.io/vatex-website/about.html), 22703 video clips, 227030 precise short captions (human annotated). Note: all clips are 10s.",
    )
    add_dataset(osmo_vatex)

    osmo_sharegpt4v_sft = Dataset(
        dataset_name="osmo_sharegpt4v_sft",
        dataset_type="torch",
        data_path="/mnt/amlfs-01/home/fuzhaox/image_datasets/ShareGPT4V/jason-filter-sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json",
        image_path="/mnt/amlfs-01/home/fuzhaox/image_datasets/ShareGPT4V/data",
        description="Original data source: https://sharegpt4v.github.io/ 655K llava_1_5_sft data relablled w/ ShareGPT4V captioner.",
    )
    add_dataset(osmo_sharegpt4v_sft)

    osmo_sharegpt4v_gpt4_100k = Dataset(
        dataset_name="osmo_sharegpt4v_gpt4_100k",
        dataset_type="torch",
        data_path="/mnt/amlfs-01/home/fuzhaox/image_datasets/ShareGPT4V/jason-filter-sharegpt4v_instruct_gpt4-vision_cap100k.json",
        image_path="/mnt/amlfs-01/home/fuzhaox/image_datasets/ShareGPT4V/data",
        description="Original data source: https://sharegpt4v.github.io/ ~100K long Image - Text pair generated by GPT4V.",
    )
    add_dataset(osmo_sharegpt4v_gpt4_100k)
    
    osmo_vflan = Dataset(
        dataset_name="osmo_vflan",
        dataset_type="vflan",
        data_path="/mnt/amlfs-01/home/fuzhaox/image_datasets/vflan/vlm-flan-clean-text1m-nosqa",
    )
    add_dataset(osmo_vflan)

    osmo_llava_1_5_mm_align = Dataset(
        dataset_name="osmo_llava_1_5_mm_align",
        dataset_type="torch",
        data_path="/mnt/amlfs-01/home/fuzhaox/image_datasets/LLaVA-CC3M-Pretrain-595K/chat.json",
        image_path="/mnt/amlfs-01/home/fuzhaox/image_datasets/LLaVA-CC3M-Pretrain-595K/images",
    )
    add_dataset(osmo_llava_1_5_mm_align)

    osmo_activitynet_qa = Dataset(
        dataset_name="osmo_activitynet_qa",
        dataset_type="torch",
        data_path="/mnt/amlfs-01/home/fuzhaox/video_datasets_v2/video_chatgpt/video_chatgpt_caption/train-processed-filtered-v2.json",
        image_path="/mnt/amlfs-01/home/fuzhaox/video_datasets_v2/video_chatgpt/video_chatgpt_videos/activitynet_videos",
        description="28250 human-annotated QA pairs on 2825 videos derived from the popular ActivityNet dataset.",
    )
    add_dataset(osmo_activitynet_qa)
    osmo_ivqa = Dataset(
        dataset_name="osmo_ivqa",
        dataset_type="torch",
        data_path="/mnt/amlfs-01/home/fuzhaox/video_datasets_v2/ivqa/ivqa/train-processed-filtered.json",
        image_path="/mnt/amlfs-01/home/fuzhaox/video_datasets_v2/ivqa/ivqa/video_data_clipped",
        description="iVQA dataset, 5378 videos with 5378 QA pairs. The 5378 QA pairs are from various domains.",
    )
    add_dataset(osmo_ivqa)
    osmo_nextqa = Dataset(
        dataset_name="osmo_nextqa",
        dataset_type="torch",
        data_path="/mnt/amlfs-01/home/fuzhaox/video_datasets_v2/nextqa/nextqa/train-processed.json",
        image_path="/mnt/amlfs-01/home/fuzhaox/video_datasets_v2/nextqa/nextqa/NExTVideo",
        description="NextQA dataset(https://github.com/doc-doc/NExT-QA/tree/main), 34132 human annotated questions from various domains.",
    )
    add_dataset(osmo_nextqa)
    osmo_msrvttqa = Dataset(
        dataset_name="osmo_msrvttqa",
        dataset_type="torch",
        data_path="/mnt/amlfs-01/home/fuzhaox/video_datasets_v2/msrvttqa/msr_vtt/train-processed-qa-v2.json",
        image_path="/mnt/amlfs-01/home/fuzhaox/video_datasets_v2/msrvttqa/msr_vtt/train_val_videos/TrainValVideo",
        description="6321 videos with 6321 rewritten QA-pairs based on the rewritten captions. (The typos in captions have been fixed by GPT-3.5-turbo)",
    )
    add_dataset(osmo_msrvttqa)
