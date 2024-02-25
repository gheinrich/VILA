from llava.train.SimpleCoyoDataset import SimpleCoyoDataset, COYO_25M_VILA, DATACOMP

for dpath in (COYO_25M_VILA, DATACOMP):
    dst = SimpleCoyoDataset(
        data_path=dpath)
    print(dst[0])