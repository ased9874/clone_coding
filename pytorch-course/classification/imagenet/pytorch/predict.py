import os
import argparse

import pandas as pd
from PIL import Image
import cv2 as cv
import torch

from src.dataset import get_transform
from src.model import create_model
from src.utils import set_seed


def predict(args):
    device = args.device
    image_path = args.image_path
    model_name = args.model_name
    
    image = Image.open(os.path.join(image_path)).convert('RGB')          #이미지 불러오기   #이미지 로딩 및 전처리(PIL + transform)
    transform = get_transform(state = 'predict', image_size = 256)       # torchvision transforms 적용
    pred_image = transform(image).unsqueeze(0).to(device) #배치 추가
    
    model = create_model(model = model_name).to(device)                           #모델 생성 및 가중치 로딩(create_model)
    model.load_state_dict(torch.load(f'best_epoch-imagenet-{model_name}.pth'))
    
    model.eval()             #예측 수행
    with torch.no_grad():
        pred = model(pred_image)
        pred_cls = pred[0].argmax(0)      #pred[0] : 배치 사이즈에 따라 0번째 이미지에 대한 각 클래스별 점수 #argmax(0): 이 벡터값에서 가장 큰 인덱스 찾기
           
    txt_path = '../dataset/folder_num_class_map.txt'
    classes_map = pd.read_table(txt_path, header = None, sep= ' ') # 클래스 번호 ↔ 이름 연결된 txt 읽기
    classes_map.columns = ['folder', 'number', 'classes']        
    
    pred_label = classes_map['classes'][pred_cls.item()] # 예측된 클래스 이름 추출
    cv_image = cv.imread(image_path)
    cv_image = cv.resize(cv_image, (800,600))                                                  #OpenCV로 이미지에 예측 결과 텍스트 표시
    cv.putText(   # 이미지 위에 예측 결과 쓰기
        cv_image,
        f'Predicted class: "{pred_cls[0]}", Predicted label: "{pred_label}"',
        (50,50),
        cv.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,0,0),
        2
    )
    
    cv.imshow('Predicted Image', cv_image)
    cv.waitKey(0) #키보드 입력을 기다리며 무한 대기(유저가 키를 누를 때까지 기다림)
    cv.destroyAllWindows()# 모든 창을 닫는 함수
    
    
    if __name__ == '__main__': #<- 스크립트 실행 시 시작점
        set_seed(36)  # seed 고정 
        
        parser = argparse.ArgumentParser()  #argparse로 실행 인자 받기
        parser.add_argument("--device", default = 'cpu', help = '학습에 사용되는 장치')
        parser.add_argument("--model", dest = 'model_name', default = 'efficientnet', help = '학습에 사용되는 모델')
        parser.add_argument("--image_path", type = str, help = "예측할 이미지 선택")
        args = parser.parse_args() #parse_args :사용자가 입력한 parser를 처리하겠다 
        
        predict(args)  #predict 실행
        
    
    