import os
import glob
import random

import torch 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split  #사이킷런에서 

def rename_dir(txt_path: os.PathLike, data_dir: os.PathLike):
    classes_map = pd.read_table(txt_path, header = None, sep = ' ')
    classes_map.columns = ['folder','number','classes']
    class_dict = {}
    for i in range(len(classes_map)):#len(classes_map): 데이터프레임의 행의 수 
        class_dict[classes_map['folder'][i]] = f'{classes_map["number"][i]-1}-{classes_map["classes"][i]}' #기존 볼더명을 새 이름으로 바꿈
    for dir, cls in class_dict.items():
        src = os.path.join(data_dir, dir)
        dst = os.path.join(data_dir,cls)
        try:
            os.rename(src,dst)
        except:
            pass
        
    return class_dict

def split_dataset(data_dir: os.PathLike, split_rate: float = 0.2) -> None: # split_rate: train과 test 얼마나 나눌건지
    image_dir = glob.glob(f'{data_dir}/*/*.JPEG')
    class_names = list(map(lambda path: os.path.basesname(os.path.dirname(path)),image_dir))
    train_x, val_x, train_y,val_y = train_test_split(image_dir,class_names,test_size =split_rate, stratify = class_names)  #stratify(계층화하다) : class_names의 비율를 유지해서 나누다.
    val_x, test_x, val_y, test_y = train_test_split(val_x, val_y, test_size = 0.5, stratify = val_y)  # -> 검증데이터를 다시 반으로 잘라서 test데이터 만들기
    
    return train_x, train_y, val_x, val_y, test_x, test_y


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 
