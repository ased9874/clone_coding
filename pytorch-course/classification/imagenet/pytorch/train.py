import argparse

import wandb #Weights & Biases (W&B, 가중치와 편향) : 머신러닝 프로젝트의 실험 과정을 추적하고 시각화아며 협업을 돕는 개발자 도구 
import torch
from torch import nn, optim #신경망 모델을 만들고 모델 최적화할 때 사용
from torch.utils.data import DataLoader # 데이터를 효율적으로 로드할 때 사용

from src.dataset import ImagenetDataset, get_transform 
from src.model import create_model
from src.utils import rename_dir, split_dataset, set_seed # 보조 함수들


# 이미지 분류 모델을 훈련하고 검증하며, wandb으로 과정 추적
def main(args):
    wandb.init(project = "imagenet-classification")  # wandb.init : 세션을 시작하고 모든 로그를 "imagenet-classification"의 프로젝트에 기록
    
    batch_size = args.batch_size
    epochs = args.epochs
    lr = 1e-3 #learning rate : 학습률 
    device = args.device
    image_size = args.image_size
    model_name = args.model_name
    
    data_dir = '../dataset' # 현재 실행 중인 스크립트 파일 위치 기준으로 상위 폴더에 있는 데이터 셋을 가르킴
    class_txt = '../dataset/folder_num_class_map.txt'
    _ = rename_dir(txt_path = class_txt, data_dir = data_dir)
    train_x, train_y, val_x, val_y, _, _ = split_dataset(data_dir= data_dir)
    
    train_data = ImagenetDataset(
        image_dir = train_x,
        class_name = train_y,
        transform = get_transform(image_size = image_size)
    )
    valid_data = ImagenetDataset(
        image_dir = val_x,
        class_name = val_y,
        transform = get_transform(image_size = image_size)
    )
    print("Initialize Dataset\n")
    
    train_dataloader = DataLoader(train_data, batch_size = batch_size, num_workers = 0)
    valid_dataloader = DataLoader(valid_data, batch_size = batch_size, num_workers = 0)
    print("Initialize DataLoader\n")
    
    model = create_model(model = model_name).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = lr, momentum = 0.9) #model.parameters(): 모델이 학습할 모든 가중치와 편향을 이터레이터 형태로 변환한 것
    print("Initialize Pytorch Model\n")                                ## momentum=0.9: SGD 옵티마이저에 '모멘텀'이라는 기술을 적용하여 학습 안정성과 속도를 높임
    
    best_accuracy = 0
    
    for t in range(epochs):
        print(f'Epoch {t+1}\n-------------------------------')
        # training
        size = len(train_dataloader.dataset)
        model.train()
        for batch, (image,targets) in enumerate(train_dataloader):
            images = images.to(device)
            targets = targets.to(device)
            targets = torch.flatten(targets)
            
            preds = model(images)
            loss = loss_fn(preds, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch % 100 == 0: #100 배치마다 현재 훈련 손실을 출력합니다.
                loss = loss.item()
                current = batch * len(images)
                print(f'loss: {loss:>7f} [{current:>5d}/{size:>5d}]')
                
            wandb.log({  #W&B에 현재 훈련 손실을 기록
                "epoch": t + 1,
                "train_loss": loss,
            })
            
        # validation  검증단계
        size = len(valid_dataloader.dataset) # 전체 검증 데이터셋의 이미지 개수
        num_batches = len(valid_dataloader)  # 검증 데이터로더가 가지고 있는 배치(묶음)의 총 개수
        model.eval()
        total_valid_loss = 0  # 전체 검증 손실 누적 변수
        correct = 0  # 이미지 맞힌 개수 누적 변수
        with torch.no_grad():
            for images, targets in valid_dataloader:  # 검증 데이터 로더에서 배치 단위로 이미지와 정답 가지고 오기
                images = images.to(device)
                targets = targets.to(device)
                targets = torch.flatten(targets)
                
                preds = model(images)
                valid_loss = loss_fn(preds, targets)
                
                total_valid_loss += valid_loss.item()
                correct += (preds.argmax(1) == targets).float().sum().item()
                
                wandb.log({   # wandb에 현재 검증 손실을 기록 
                    "epoch": t + 1,  #몇 번째 에포크인지
                    "val_loss": valid_loss.item(), # 로그 값 
                })
        
        total_valid_loss /= num_batches  # 전체 손실 누적/ 배치 수 = 평균 검증 손실
        correct /= size # 검증 정확도(비율 계산)
        print(f'Valid Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {total_valid_loss:>8f} \n')
        
        wandb.log({
            "epoch": t + 1,
            "valid_accuracy": correct * 100
        })
        
        if best_accuracy < correct: # 현재 에포크 검증 정확도가 지금까지의 최고 정확도보다 높으면 
            best_accuracy = correct # 업데이트
            torch.save(model.state_dict(), f'best_epoch-imagenet-{model_name}.pth') # 모델의 현재 상태(가중치, 편향)를 파일로 저장(.pth) 확장자
            print(f'{t+1} epoch: Saved Model State to best_epoch-imagenet-{model_name}.pth\n')
            
    torch.save(model.state_dict(), f'last_epoch-imagenet-{model_name}.pth')
    print(f'Saved Model State to last_epoch-imagenet-{model_name}.pth\n')
    
    print('Done!')
    wandb.finish()
    
    
if __name__ == '__main__':  #__main__ : 이 스크립트가 직접 실행될 때 내장 변수__name__에 __main__ 으로 저장됨  #"만약 이 파이썬 스크립트가 메인 프로그램으로 직접 실해되고 있다면"
    set_seed(36)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type =str, default = "cpu", help = "학습에 사용되는 장치")
    parser.add_argument("--image_size", type = int, default = 256, help = "이미지 resize 크기")
    parser.add_argument("--batch_size", type = int, default = 64, help = "훈련 배치 사이즈 크기")
    parser.add_argument("--epochs", type = int, default = 30, help = "훈련 에폭 크기")
    parser.add_argument("--model", dest = 'model_name', type =str, default = 'efficientnet',help = "사용할 모델 선택")
    args = parser.parse_args()
    
    main(args)
            