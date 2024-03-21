import json
import os
import signal
import subprocess
from concurrent.futures import ThreadPoolExecutor

import decord
# from decord import VideoReader
import numpy as np
import torch
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms import Resize

decord.bridge.set_bridge("torch")

video_json_path = "/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets_v2/vatex/vatex_training_processed_filtered.json"
cache_video_jsonl_path = "/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets_v2/vatex/vatex_training_processed_filtered-v2.jsonl"
video_path_out = "/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets_v2/vatex/vatex_training_processed_filtered.json"
video_dir = "/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets_v2/vatex/videos_clipped"


SKIP_ID = [
    "7olz18uNTUI_000286_000296",
    "Bk0i4M38dLk_000257_000267"
    # "tQ6-_e59Zrk_1"
    # "v_1296743-2", "v_909502", "v_977136", "v_985798", "v_954878", "v_985376",
    # "v_977995", "v_1409590", "v_985116", "v_959350-2", "v_981773", "v_1232351",
    # "v_988656", "v_983727", "v_954159"
]
VIDEO_LIST = [
    "/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets_v2/vatex/videos_clipped/kI4W37Ipwds_000085_000095.mp4",
    "/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets_v2/vatex/videos_clipped/xcxcGeFv0i0_000136_000146.mp4",
    "/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets_v2/vatex/videos_clipped/CaXa9TDy4S8_000044_000054.mp4",
    "/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets_v2/vatex/videos_clipped/MK_qkoBBo38_000010_000020.mp4",
    "/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets_v2/vatex/videos_clipped/KpDUUFBYs6U_000093_000103.mp4",
    "/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets_v2/vatex/videos_clipped/FOoHKpr1xs8_000002_000012.mp4",
    "/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets_v2/vatex/videos_clipped/FRei3a5Gqio_000003_000013.mp4",
    "/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets_v2/vatex/videos_clipped/q7jhDND8xjA_000155_000165.mp4",
    "/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets_v2/vatex/videos_clipped/bfrdgr9G8-g_000032_000042.mp4",
    "/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets_v2/vatex/videos_clipped/PmLHRVo4dP0_000050_000060.mp4",
]


# Read jsonl file if avaible
cache_video_json = []
cache_video_ids = []
if os.path.exists(cache_video_jsonl_path):
    with open(cache_video_jsonl_path, "r") as f:
        for line in f:
            cache_video_json.append(json.loads(line))
for video in cache_video_json:
    cache_video_ids.append(video["id"])


def load_video(video_path, num_video_frames):
    # video_reader = VideoReader(uri=video_path)
    image_size = 384

    # idx = np.round(np.linspace(0, len(video_reader) - 1, num_video_frames)).astype(int)
    try:
        # video_path = "/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets_v2/youcook2/video_data_clipped/FtHLUsOntqI_5.mp4"
        video = EncodedVideo.from_path(video_path, decoder="decord", decode_audio=False)
        duration = float(video.duration)
        assert duration >= 0.5
        video_outputs = video.get_clip(start_sec=0, end_sec=duration)["video"]
        assert video_outputs.size(1) > 8
        print(f"video_outputs.size(1) {video_outputs.size(1)}")
        indices = torch.linspace(0, video_outputs.size(1) - 1, 8).long()
        # print(f'indices {indices}')
        video_outputs = video_outputs[:, indices, :, :]
        # print(f'video_outputs new size {video_outputs.size()}')
    except Exception as e:
        print(f"bad data path {video_path}")
        print(f"Error processing {video_path}: {e}")
        video_outputs = torch.zeros(3, 8, image_size, image_size, dtype=torch.uint8)

    c, b, h, w = video_outputs.size()
    image_tensor = torch.zeros(b, c, image_size, image_size, dtype=torch.uint8)
    video_frames = video_outputs.permute(1, 0, 2, 3).contiguous()
    video_frames = Resize(size=[image_size, image_size], antialias=True)(video_frames)
    image_tensor[:, :, :, :] = video_frames

    return image_tensor


def test_valid_video():
    video_json = json.load(open(video_json_path, "r"))
    video_json_output = []
    processed_files = 0
    for video in video_json:
        print(f"Processing {processed_files} files")
        print(f"{video}")
        processed_files += 1
        # if video['id'] in cache_video_ids or video['id'] in SKIP_ID:
        #     continue
        if "video" in video.keys():
            path = os.path.join(video_dir, video["video"])
        else:
            path = os.path.join(video_dir, video["id"] + ".mp4")
        print(f"Processing {path}")
        video_data = load_video(path, 8)
        del video_data
        print(f"Processed {path}")
        video_json_output.append(video)
        if processed_files % 10 == 0:
            with open(cache_video_jsonl_path, "a") as f:
                for video in video_json_output:
                    f.write(json.dumps(video) + "\n")
            video_json_output = []
    json.dump(video_json_output, open(video_path_out, "w"))


for vid_path in VIDEO_LIST:
    video_data = load_video(vid_path, 8)
# load_video('/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets_v2/vatex/videos_clipped/fa1WrHOTjxY_000512_000522.mp4', 8)
# test_valid_video()

# # Write cache to json file
# with open(video_path_out, "w") as f:
#     json.dump(cache_video_json, f)
