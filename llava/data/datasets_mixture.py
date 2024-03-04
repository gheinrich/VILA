from dataclasses import dataclass, field


@dataclass
class Dataset:
    dataset_name: str
    dataset_type: str = field(default="torch")
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    image_path: str = field(default=None, metadata={"help": "Path to the training image data."})
    description: str = field(
        default=None,
        metadata={
            "help": "Detailed desciption of where the data is from, how it is labelled, intended use case and the size of the dataset."
        },
    )


DATASETS = {}

import warnings 
def add_dataset(dataset):
    if dataset.dataset_name in DATASETS:
        # make sure the data_name is unique
        warnings.warn(f"{dataset.dataset_name} already existed in DATASETS. Make sure the name is unique.")
    assert "+" not in dataset.dataset_name, "Dataset name cannot include symbol '+'."
    DATASETS.update({dataset.dataset_name: dataset})


def register_datasets_mixtures():
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

    # data_path='/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/datasets/captioning/coyo-700m_full_webdata',
    # data_path='/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/datasets/captioning/coyo-25m-vila',
    coyo_25m_wds = Dataset(
        dataset_name="coyo_25m_wds",
        dataset_type="coyo-wds",
        data_path="/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/dataset/coyo-25m-vila",
        description="See coyo. Convert coyo to webds format.",
    )
    add_dataset(coyo_25m_wds)
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
        data_path="/home/jasonlu/datasets/coyo-700m/pkl02-split",
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

    mmc4core = Dataset(
        dataset_name="mmc4core",
        dataset_type="mmc4",
        # data_path='/home/jasonlu/datasets/mmc4-core/pkl-core',
        data_path="/home/jasonlu/datasets/mmc4-core/pkl-core",
        description="Original data source: https://github.com/allenai/mmc4 mmc4-core that contains 29.9M images, interleaved Image - Text data.",
    )
    add_dataset(mmc4core)

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
        data_path="/home/jasonlu/datasets/ccs_recaptioned",
        description="TODO dannyy",
    )
    add_dataset(ccs_recaptioned)

    ccs_recaptioned_test = Dataset(
        dataset_name="ccs_recaptioned_test",
        dataset_type="wds",
        data_path="/home/jasonlu/datasets/ccs_recaptioned_test",
        description="See ccs_recaptioned, A subset of ccs_recaptioned (16 shards) that could be used for test purposes.",
    )
    add_dataset(ccs_recaptioned_test)

    vflan = Dataset(
        dataset_name="vflan",
        dataset_type="vflan",
        data_path="/home/jasonlu/datasets/vlm-flan-clean-text1m-nosqa",
    )
    add_dataset(vflan)

    llava_1_5_mm_align = Dataset(
        dataset_name="llava_1_5_mm_align",
        dataset_type="torch",
        data_path="./playground/data/LLaVA-Pretrain/LLaVA-CC3M-Pretrain-595K.json",
        image_path="./playground/data/LLaVA-Pretrain/images",
    )
    add_dataset(llava_1_5_mm_align)
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
        data_path="/home/jasonlu/vlm_datasets/ShareGPT4V/sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json",
        image_path="/home/jasonlu/vlm_datasets/ShareGPT4V/data",
        description="Original data source: https://sharegpt4v.github.io/ 655K llava_1_5_sft data relablled w/ ShareGPT4V captioner.",
    )
    add_dataset(sharegpt4v_sft)
    sharegpt4v_gpt4_100k = Dataset(
        dataset_name="sharegpt4v_gpt4_100k",
        dataset_type="torch",
        data_path="/home/jasonlu/vlm_datasets/ShareGPT4V/filter-sharegpt4v_instruct_gpt4-vision_cap100k.json",
        image_path="/home/jasonlu/vlm_datasets/ShareGPT4V/data",
        description="Original data source: https://sharegpt4v.github.io/ ~100K long Image - Text pair generated by GPT4V.",
    )
    add_dataset(sharegpt4v_gpt4_100k)
    
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
    
    ai2d = Dataset(
        dataset_name="ai2d",
        dataset_type="torch",
        data_path="/home/yunhaof/workspace/datasets/AI2D/processed/train_12k.json",
        image_path="/home/yunhaof/workspace/datasets/AI2D/ai2d/images",
    )
    add_dataset(ai2d)
    
    synthdog_en = Dataset(
        dataset_name="synthdog_en",
        dataset_type="torch",
        data_path="/home/yunhaof/workspace/datasets/synthdog-en/synthdog_en_66_5k_with_question.json",
        image_path="/home/yunhaof/workspace/datasets/synthdog-en/images",
    )
    add_dataset(synthdog_en)
    
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
        data_path="/home/yunhaof/workspace/datasets/GRIT/processed-grit-2m/grit_merged_qas_1613k.json",
        image_path="/home/yunhaof/workspace/datasets/GRIT/processed-grit-2m/webdataset_untar",
    )
    add_dataset(grit_mixture)
    
    sharegpt4v_pretrain = Dataset(
        dataset_name="sharegpt4v_pretrain",
        dataset_type="torch",
        data_path="/home/jasonlu/vlm_datasets/ShareGPT4V/filter-share-captioner_coco_lcs_sam_1246k_1107.json",
        image_path="/home/jasonlu/vlm_datasets/ShareGPT4V/data",
        description="Original data source: https://sharegpt4v.github.io/ ~1M long Image - Text pair generated by ShareGPT4V captioner.",
    )
    add_dataset(sharegpt4v_pretrain)
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
        data_path="/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets_v2/jukinmedia/jukin-100k-processed-long-filtered.json",
        image_path="/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets_v2/jukinmedia/videos",
        description="A high quailty video caption dataset with 71003 detailed captions (16 words at least).",
    )
    add_dataset(jukinmedia)
    youcook2 = Dataset(
        dataset_name="youcook2",
        dataset_type="torch",
        data_path="/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets_v2/youcook2/youcookii_clipped-v2.json",
        image_path="/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets_v2/youcook2/video_data_clipped",
        description="YouCook2 (http://youcook2.eecs.umich.edu/): A large-scale video dataset with 11680 short but precise human written captions.",
    )
    add_dataset(youcook2)
    vatex = Dataset(
        dataset_name="vatex",
        dataset_type="torch",
        data_path="/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets_v2/vatex/vatex_training_processed_filtered-v2.json",
        image_path="/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets_v2/vatex/videos_clipped",
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

    # Video Pretraining Datasets added by Fuzhao
    internvid_test = Dataset(
        dataset_name="internvid_test",
        dataset_type="video-wds",
        data_path="/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets_v2/internvid/video_data_tar/InternVid-8K-flt",
        # cache_path='/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets_v2/internvid/video_data_tar/InternVid-8K-flt-webds-meta',
        description="A tiny debug set of internvid with only 8K samples.",
    )
    add_dataset(internvid_test)
    internvid_1300K = Dataset(
        dataset_name="internvid_1300K",
        dataset_type="video-wds",
        data_path="/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets_v2/internvid/video_data_tar/InternVid-1300K-flt",
        # cache_path='/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets_v2/internvid/video_data_tar/InternVid-1300K-flt-webds-meta',
        description="1M (not 1300K after cleaning) video-caption pairs from InternVid. We select the top-relevant 1M samples from the Intern-Vid-10M set.",
    )
    add_dataset(internvid_1300K)
