import os

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset # coco도 커스텀 데이터셋이므로 pytorch.DataLoader 사용하려면 Dataset을 상속 받아야 함
from pycocotools.coco import COCO


class COCODataset(Dataset):
    def __init__(self,image_dir, annotation_file, image_list, transform):
        self.coco = COCO(annotation_file) # COCO 형식의 JSON 주석 파일 경로를 전달하여 COCO 객체를 초기화한다. 주석을 가지고 와서 구조화된 형식으로 파싱하여 쉽게 검색할 수 있도록 함
        self.image_dir = image_dir #이미지 저장 루트
        self.image_list = image_list #대아토샛애 포함하려는 이미지들의 전체 경로 리스트
        self.transform = transform
        
    def __len__(self):
        return len(self.image_list) #이미지 개수 반환
    
    def __getitem__(self, index):
        image_path = self.image_list[index]  #index 에 해당하는 이미지 전체 경로
        file_name = os.path.basename(image_path)  #basename : 파일 이름만 추출
        
        for img in self.coco.dataset['images']: 
            if img['file_name'] ==file_name:
                image_id = img['id'] # image_id: COCO 데이터셋 내에서 각 이미지 고유 번호
                
        image = Image.open(image_path).convert("RGB") #
        original_width, original_height = image.size  #이미지 원본 크기 저장
        
        ann_ids = self.coco.getAnnIds(imgIds = image_id)
        annotations = self.coco.loadAnns(ann_ids)
        
        boxes = []  #처리된 bbox 좌표와
        labels = [] # 해당 카테고리 레이블 저장 리스트 초기화
        
        # bbox 좌표와 카테고리 ID 추출
        for ann in annotations:   
            x,y,w,h = ann['bbox'] 
            boxes.append([x,y,x+w, y+h])
            labels.append(ann['category_id'])
            
            
        if len(boxes) == 0:  # 주석(bbox) 가 없는 경우 : 비어 있는 텐서 반환
            boxes = torch.zeros((0,4), dtype = torch.float32)
            labels = torch.zeros((0,), dtype = torch.int64)
            
        else:
            boxes = torch.tensor(boxes, dtype = torch.float32)
            labels = torch.tensor(labels, dtype = torch.int64)
            
        if self.transform:
            image = self.transform(image)
            new_width, new_height = 256,256
            
            scale_x = new_width / original_width
            scale_y = new_height / original_height
            
            boxes[:, [0,2]] *= scale_x  #너비와 높이에 대한 스케일링 비율을 계산합니다.
            boxes[:, [1,3]] *= scale_y
            
            boxes[:, [0,2]] = boxes[:, [0,2]].clamp(0, new_width)  #클램핑 : 스케일링 후에 어떤 좌표값도도 새 이미지 크기보다 작게 값을 제한
            boxes[:, [1,3]] = boxes[:, [1,3]].clamp(0, new_height)
        # 출력 레이블  
        target = {  
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([image_id])
        }
        
        return image, target
        
    
    