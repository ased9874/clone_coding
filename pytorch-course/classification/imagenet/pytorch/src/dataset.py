import os

from typing import Callable, Optional, Sequence, Tuple
from PIL import Image

import torch
from torch.utils.data import Dataset  # Dataset을 상속 받아야 하는 이유. 파이토치에서 사용자 정의 데이터셋을 만들 때 반드시 상속되야 Dataloader를 쓸 수 있다 
from torchvision import transforms    #Dataset을 상속 받으면 1. __len__ 메서드 구현:데이터세의 총 샘플 개수를 반환 2. getitem 메서드 구현 :특정 인덱스에 해당하는 샘플 하나와 레이블 반환

class ImagenetDataset(Dataset):      #image_dir 인자는 str이나 Path 객체처럼 경로(path)처럼 사용할 수 있는 값,    #image_dir는 경로 한개가 아닌 여러 경로의 리스트이디ㅏ 
                                      #os.PathLike 는 "images/cat/0001.jpg"나 Path("images/cat/0001.jpg") 모두 허용
    def __init__(self, image_dir: os.PathLike, class_name: str, transform: Optional[Sequence[Callable]]) -> None:# 아무것도 반환하지 않는단 의미, 타입오류를 잘 잡아냄. 사용자가 이해하기 편함
        super().__init__()
        self.image_dir = image_dir
        self.class_name = class_name
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.image_dir)
    
    def __getitem__(self, index: int) -> Tuple:
        image_id = self.image_dir[index]
        target = self.class_name[index].split("-")[0]
        
        image = Image.open(os.path.join(image_id).convert('RGB')) #이미지를 열고 단일 채널에서 색상 채널로 변환
        image = self.transform(image)
        
        target = torch.tensor(int(target))
        
        return image, target
    
    def get_transform(image_size: int =256):
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229,0.224,0.225)) #이미지 픽셀값을 평균 0, 표준편차 1로 정규화
                                              #ImageNet 데이터셋의 RGB 평균/표준편차 
        ])
        
        return transform