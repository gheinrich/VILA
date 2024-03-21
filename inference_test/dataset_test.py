from llava.train.SimpleCoyoDataset import COYO_25M_VILA, DATACOMP, SimpleCoyoDataset

for dpath in (COYO_25M_VILA, DATACOMP):
    dst = SimpleCoyoDataset(data_path=dpath)
    print(dst[0])
