from dataclasses import dataclass, field

@dataclass
class Dataset:
    dataset_name: str
    dataset_type: str = field(default="torch")
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    image_path: str = field(
        default=None, metadata={"help": "Path to the training image data."}
    )

DATASETS_MIXTURES = {}
DATASETS = {}

def add_dataset(dataset):
    DATASETS.update({dataset.dataset_name: dataset})


def register_datasets_mixtures():
    datacomp_webds = Dataset(
        dataset_name='datacomp_webds',
        dataset_type='coyowebds',
        data_path="/lustre/fsw/portfolios/llmservice/users/dannyy/dannyy_gpt4/data_filtering/dc1b_filtered",
    )
    
    
    coyo_webds_refilerted = Dataset(
        dataset_name='coyo_webds_refilerted',
        dataset_type='coyowebds',
        data_path="/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/datasets/captioning/coyo-700m_full_webdata_fullmeta/stage2_filtered_v2",
    )
    add_dataset(coyo_webds_refilerted)
    
    
    coyo_webds_vila_recaption = Dataset(
        dataset_name='coyowebds_vila_recaption',
        dataset_type='coyowebds_recap',
        data_path="/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/datasets/captioning/coyo-25m-vila",
    )
    add_dataset(coyo_webds_vila_recaption)
    
    coyo_webds_vila = Dataset(
        dataset_name='coyowebds',
        dataset_type='coyowebds',
        # data_path='/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/datasets/captioning/coyo-700m_full_webdata',
        data_path="/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/datasets/captioning/coyo-25m-vila",
    )
    add_dataset(coyo_webds_vila)
    coyo_webds_full = Dataset(
        dataset_name='coyowebds_full',
        dataset_type='coyowebds',
        data_path='/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/datasets/captioning/coyo-700m_full_webdata',
    )
    add_dataset(coyo_webds_full)
    coyo_25m = Dataset(
        dataset_name='coyo',
        dataset_type='coyo',
        data_path='/home/jasonlu/datasets/coyo-700m/pkl02-split')
    add_dataset(coyo_25m)
    coyo_25m_test = Dataset(
        dataset_name='coyo_test',
        dataset_type='coyo',
        data_path='/home/jasonlu/vlm_datasets/debug/coyo-700m/pkl02-split')
    add_dataset(coyo_25m_test)
    mmc4core = Dataset(
        dataset_name='mmc4core',
        dataset_type='mmc4sub',
        data_path='/home/jasonlu/datasets/mmc4-core/pkl-core')
    add_dataset(mmc4core)
    mmc4core_test = Dataset(
        dataset_name='mmc4core_test',
        dataset_type='mmc4sub',
        data_path='/home/jasonlu/vlm_datasets/debug/mmc4-core/pkl-core')
    add_dataset(mmc4core_test)
    ccs_recaptioned = Dataset(
        dataset_name='ccs_recaptioned',
        dataset_type='wds',
        data_path='/home/jasonlu/datasets/ccs_recaptioned')
    add_dataset(ccs_recaptioned)
    ccs_recaptioned_test = Dataset(
        dataset_name='ccs_recaptioned_test',
        dataset_type='wds',
        data_path='/home/jasonlu/datasets/ccs_recaptioned_test')
    add_dataset(ccs_recaptioned_test)
    vflan = Dataset(
        dataset_name='vflan',
        dataset_type='vflan',
        data_path='/home/jasonlu/datasets/vlm-flan-clean-text1m-nosqa')
    add_dataset(vflan)
    laion = Dataset(
        dataset_name='laion',
        dataset_type='coyo',
        data_path='/home/jasonlu/datasets/laion-recap-pkl')
    add_dataset(laion)
    llava_1_5_sft = Dataset(
        dataset_name='llava_1_5_sft',
        dataset_type='torch',
        data_path='/home/jasonlu/datasets/llava-1.5/llava_v1_5_mix665k.json',
        image_path='/home/jasonlu/vlm_datasets/ShareGPT4V/data'
    )
    add_dataset(llava_1_5_sft)
    sharegpt4v_sft = Dataset(
        dataset_name='sharegpt4v_sft',
        dataset_type='torch',
        data_path='/home/jasonlu/vlm_datasets/ShareGPT4V/sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json',
        image_path='/home/jasonlu/vlm_datasets/ShareGPT4V/data'
    )
    add_dataset(sharegpt4v_sft)
    sharegpt4v_gpt4_100k = Dataset(
        dataset_name='sharegpt4v_gpt4_100k',
        dataset_type='torch',
        data_path='/home/jasonlu/vlm_datasets/ShareGPT4V/filter-sharegpt4v_instruct_gpt4-vision_cap100k.json',
        image_path='/home/jasonlu/vlm_datasets/ShareGPT4V/data'
    )
    add_dataset(sharegpt4v_gpt4_100k)
    sharegpt4v_pretrain = Dataset(
        dataset_name='sharegpt4v_pretrain',
        dataset_type='torch',
        data_path='/home/jasonlu/vlm_datasets/ShareGPT4V/filter-share-captioner_coco_lcs_sam_1246k_1107.json',
        image_path='/home/jasonlu/vlm_datasets/ShareGPT4V/data'
    )
    add_dataset(sharegpt4v_pretrain)
    valley = Dataset(
        dataset_name='valley',
        dataset_type='torch',
        data_path='/home/jasonlu/video_datasets/Webvid/chat.json',
        image_path='/home/jasonlu/video_datasets/Webvid/data/videos/'
    )
    add_dataset(valley)
    video_chatgpt = Dataset(
        dataset_name='video_chatgpt',
        dataset_type='torch',
        data_path='/home/jasonlu/video_datasets/Video_ChatGPT/VideoInstruct-100K/filtered_VideoInstruct100K.json',
        image_path='/home/jasonlu/video_datasets/Video_ChatGPT/activitynet_videos/'
    )
    add_dataset(video_chatgpt)
    valley_test = Dataset(
        dataset_name='valley_test',
        dataset_type='torch',
        data_path='/home/jasonlu/video_datasets/Webvid/chat_test.json',
        image_path='/home/jasonlu/video_datasets/Webvid/data/videos/'
    )
    add_dataset(valley_test)

    # TODO: 
    #   datacomp
    #   datacomp + mmc4core
    DATASETS_MIXTURES.update({'coyo_25m_refilter+mmc4core': [coyo_webds_refilerted, mmc4core]})
    DATASETS_MIXTURES.update({'coyo_25m_recap+mmc4core': [coyo_webds_vila_recaption, mmc4core]})
    
    DATASETS_MIXTURES.update({'datacomp_webds': [datacomp_webds, ]})
    DATASETS_MIXTURES.update({'coyo_webds_refilerted': [coyo_webds_refilerted, ]})
    DATASETS_MIXTURES.update({'coyo_webds_vila_recap': [coyo_webds_vila_recaption, ]})
    DATASETS_MIXTURES.update({'coyo_webds_vila': [coyo_webds_vila, ]})
    DATASETS_MIXTURES.update({'coyo_webds_full': [coyo_webds_full, ]})
    

    DATASETS_MIXTURES.update({'ccs_recaptioned': [ccs_recaptioned]})
    DATASETS_MIXTURES.update({'ccs_recaptioned_test': [ccs_recaptioned_test]})
    
    # original VILA step-1
    DATASETS_MIXTURES.update({'coyo_25m_mmc4core': [coyo_25m, mmc4core]})

    DATASETS_MIXTURES.update({'coyo_webds_vila_mmc4core_sharegpt4v': [coyo_webds_vila, mmc4core, sharegpt4v_pretrain]})

    DATASETS_MIXTURES.update({'coyo_25m_mmc4core_sharegpt4v': [coyo_25m, mmc4core, sharegpt4v_pretrain]})
    DATASETS_MIXTURES.update({'coyo_25m_mmc4core_sharegpt4v_valley': [coyo_25m, mmc4core, sharegpt4v_pretrain, valley]})
    DATASETS_MIXTURES.update({'coyo_25m_mmc4core_sharegpt4v_valley_test': [coyo_25m_test, mmc4core_test, sharegpt4v_pretrain, valley_test]})
    DATASETS_MIXTURES.update({'valley_test': [valley_test]})
    DATASETS_MIXTURES.update({'video_chatgpt': [video_chatgpt]})
    DATASETS_MIXTURES.update({'coyo_25m_test': [coyo_25m_test, ]})
    DATASETS_MIXTURES.update({'coyo_25m_mmc4core_test': [coyo_webds_vila, mmc4core_test]})
    DATASETS_MIXTURES.update({'vflan_sharegpt4v_sft': [vflan, sharegpt4v_sft]})
    DATASETS_MIXTURES.update({'vflan_llava_1_5_sft': [vflan, llava_1_5_sft]})
    DATASETS_MIXTURES.update({'vflan_captioner': [vflan, sharegpt4v_gpt4_100k]})
    DATASETS_MIXTURES.update({'captioner': [sharegpt4v_gpt4_100k]})
    DATASETS_MIXTURES.update({'vflan_sharegpt4v_sft_valley': [vflan, sharegpt4v_sft, valley]})
    DATASETS_MIXTURES.update({'vflan_sharegpt4v_sft_valley_video_chatgpt': [vflan, sharegpt4v_sft, valley, video_chatgpt]})
    