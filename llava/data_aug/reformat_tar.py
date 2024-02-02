from fire import Fire
from tqdm import tqdm
import os, os.path as osp
import tarfile
import shutil

def main(
    src_tar="~/workspace/sa_000000.tar", 
    src_folder="~/workspace",
    tgt_folder="~/workspace/dev-123",
    overwrite=False,
):
    
    src_tar_path = osp.expanduser(src_tar)
    src_folder_path = osp.expanduser(src_folder)
    tgt_folder_path = osp.expanduser(tgt_folder)
    rpath = osp.relpath(src_tar_path, src_folder_path)
    fpath = osp.join(tgt_folder_path, rpath)
    fpath_tmp = osp.join(tgt_folder_path, rpath + ".tmp")
    
    if osp.exists(fpath) and not overwrite:
        print("Skipping")
        return 
        
    
    t = tarfile.open(src_tar_path)

    os.makedirs(osp.dirname(fpath_tmp), exist_ok=True)
    tdev = tarfile.open(fpath_tmp, "w")

    for idx, member in tqdm(enumerate(t.getmembers())):
        print(idx, member, flush=True)
        tdev.addfile(member, t.extractfile(member.name))
        
    t.close()
    tdev.close()
    
    shutil.move(fpath_tmp, fpath)
    print("Finish")
    
if __name__ == "__main__":
    Fire(main)