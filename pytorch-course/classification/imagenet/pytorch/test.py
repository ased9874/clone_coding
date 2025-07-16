import argparse

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.dataset import ImagenetDataset, get_transform
from src.model import create_model
from src.utils import set_seed, split_dataset


def test(args):
    device = args.device
    model_name = args.model_name
    image_size = args.image_size
    batch_size = args.batch_size
    
    data_dir = '../dataset'  #데이터셋이 저장된 폴더 경로  #..: 부모디렉토리를 의미 # .(점 하나) : 현재 디렉토리(스크립트가 실행되는 곳)
    _, _, _, _, test_x, test_y = split_dataset(data_dir = data_dir) # 여기서는 테스트 데이터 (test_x, test_y)만 필요하므로 앞의 네 개는 _ 로 무시해라 
    
    test_data = ImagenetDataset(
        image_dir = test_x,  
        class_name = test_y, 
        transform = get_transform(state = 'test', image_size = image_size)  # state를 적는 이유? 
    )
    
    test_dataloader = DataLoader(test_data, batch_size = batch_size, num_workers = 0)  #test_data를 데이터 로더에 놓기 # num_workers >0 : cpu코어 수로 설정하면 여러개의 서브 프로세스를 생성하여 데이터를 동시에 불러옴.(gpu가 학습하는 동안 cpu가 다음 배치의 데어터를 미리 준비하여 병목현상 줄이고 속도 높이고)
                                                                         #0이면 서브프로세스는 없고 메인 프로세스에서만 수행(순차적 데이터 로딩으로 속도 저하, 그러나 디버깅에 용이성)
    model = create_model(model = model_name).to(device)
    model.load_state_dict(torch.load(f'best_epoch-imagenet-{model_name}.pth')) #모델 불러오고 가중치 로드 
    
    loss_fn = nn.CrossEntropyLoss()  # 손실함수 정의 크로스엔트로피 (다중분류를 위한)
    
    # validation(확인) 모델 평가
    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader) #테스트 데이터로더가 가지고 있는 배치(묶음)의 총 개수
    model.eval()
    total_test_loss = 0  #전체 테스트 손실을 누적할 변수
    correct = 0  #정확하게 분류한 이미지 개수를 누적할 변수
    with torch.no_grad():
        for batch, (images, targets) in enumerate(test_dataloader): #테스트 데이터로더에서 배치 단위로 이미지와 정답을 가져옴
            images = images.to(device) # 이미지 디바이스로 보내고
            targets = targets.to(device) # 정답도 지정된 장치로 옮긴다
            targets = torch.flatten(targets) #아마 targets 에 차원은 (batch_size,1)인데 그걸 flatten 을 이용해 (batch_size,)로 바꿔 model의 출력값을 맞게 한다
            
            preds = model(images) #모델에 이미지를 넣어 예측값(logits)을 구한다. (순전파)
            test_loss = loss_fn(preds, targets)  # 예측값과 정답의 손실함수를 구한다.
            
            total_test_loss += test_loss.item() #현재 배치의 손실을 total_test_loss에 더한다.
            correct += (preds.argmax(1) == targets).float().sum().item() #모델의 출력 중 (텐서이니까 1차원에 있는 텐서(이미지에 따라 타겟값이 확률로 나타낸 차원)) 행방향으로 인덱스 찾기   
                                                                         #가장 높은 확률을 가진 인덱스를 가지고 와서 실제 정답과 비교(bool) 
            if batch % 20 == 0: #20 배치마다 현재 진행 상황 출력
                loss = test_loss.item()  #test_loss 꺼내오고
                current = batch * len(images) # 지금까지 꺼내온 이미지 개수
                print(f'loss: {loss:>7f}  [{current:>5d}/{size:>5d}]') #{current:>5d}/{size:>5d}: 현재 처리된 이미지 수/ 전체 이미지 수
                
    total_test_loss /= num_batches #전체 손실을 배치 개수로 나누어 평균 손실을 구한다. 
    correct /= size  # 전체 정확하게 맞춘 개수를 전체 이미지 개수로 나눈어 정확도(비율)를 구한다
    print(f'Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {total_test_loss:>8f} \n')
    
    
if __name__ == '__main__':   # 코드 시작점
    set_seed(36)
    
    parser = argparse.ArgumentParser()  # 인자값 파서 
    parser.add_argument("--device", default = 'cpu', help = '학습에 사용되는 장치')
    parser.add_argument("--model", dest = "model_name", default = "efficientnet", help = '학습에 사용되는 모델')
    parser.add_argument("--image_size", type = int, default =256, help = "이미지 resize크기")  #이미지 조절할 크키
    parser.add_argument("--batch_size", type = int, default =64, help="훈련 배치 사이즈 크기") # 한번의 배치 사이즈
    args = parser.parse_args()
    
    test(args)