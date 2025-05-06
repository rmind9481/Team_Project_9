# 기본 라이브러리
import torch
# 신경망 모델을 정의하는데 필요한 레이어
import torch.nn as nn
# 손실함수등을 제공하는 모듈
import torch.nn.functional as F

# 폴더구조에 따라 이미지를 불러오는 데이터셋
from torchvision.datasets import ImageFolder
# 데이터배치 단위로 처리하기 위한 유틸리티
from torch.utils.data import DataLoader, Subset
# 이미지전처리를 위한 변환 함수들을 제공
from torchvision.transforms import transforms
# 최적화 알고리즘을 제공하는 토치
import torch.optim as optim

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time  # 시간 측정을 위한 모듈 추가
import datetime  # 시간 형식화를 위한 모듈 추가

# 데이터 준비
IMG_ROOT = './data/upimage'  # 경로 수정 필요

preprocessing = transforms.Compose([
    # 이미지를 (224,224)로 하고 텐서 형태로 변환하는 전처리 과정
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 데이터셋 생성

try:
    imgDS = ImageFolder(root=IMG_ROOT, transform=preprocessing)
    # ImageFolder 작동 시
    
    # 데이터 분리
    indices = list(range(len(imgDS)))
    labels = [imgDS[i][1] for i in indices]
    
    train_indices, test_indices = train_test_split(
        indices, 
        test_size=0.2, 
        stratify=labels, 
        random_state=42
    )
    
    train_dataset = Subset(imgDS, train_indices)
    test_dataset = Subset(imgDS, test_indices)
    
    num_classes = len(imgDS.classes)
    
except:
    # CSV 파일을 사용하는 방식으로 변경
    import pandas as pd
    from PIL import Image
    import os
    from torch.utils.data import Dataset
    

    class BirdDataset(Dataset):
        def __init__(self, csv_file, img_dir, transform=None):
            self.data_info = pd.read_csv(csv_file, header=None, names=['img_path', 'upscale_img_path', 'label'])
            self.img_dir = img_dir
            self.transform = transform
            
            # label_to_idx : 레이블을 숫자로 변환하여 label_idx 라는 새로운 열을추가 
            self.label_set = sorted(list(set(self.data_info['label'])))
            self.label_to_idx = {label: idx for idx, label in enumerate(self.label_set)}
            self.data_info['label_idx'] = self.data_info['label'].apply(lambda x: self.label_to_idx[x])
            
            self.classes = self.label_set
        
        def __len__(self):
            return len(self.data_info)
        
        def __getitem__(self, idx):
            img_name = os.path.join(self.img_dir, os.path.basename(self.data_info.iloc[idx, 1]))  # upscale_img_path 사용
            image = Image.open(img_name).convert('RGB')
            label = self.data_info.iloc[idx, 3]  # label_idx 사용
            
            if self.transform:
                image = self.transform(image)
                
            return image, label
    
    # CSV 파일 및 이미지 경로 설정
    csv_file = './data/filtered_dataset.csv'  # CSV 파일 경로 수정 필요
    
    # 데이터셋 생성
    dataset = BirdDataset(csv_file=csv_file, img_dir=IMG_ROOT, transform=preprocessing)
    
    # 데이터 분리
    indices = list(range(len(dataset)))
    labels = [dataset.data_info.iloc[i]['label_idx'] for i in indices]
    
    train_indices, test_indices = train_test_split(
        indices, 
        test_size=0.2, 
        stratify=labels, 
        random_state=42
    )
    
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    
    num_classes = len(dataset.classes)

# 인공신경망 
# =========================================================================
# 1) 순전파 => 입력층 부터 출력층 까지 순서대로 변수들을 계산
# 2) 손실함수 계산 => 출력층의 결과와 실제값을 활용하여 손실함수계산
# 3) 역전파 => 출력층에서 입력층으로 역행하여 가중치 와 편향

# 역전파 => 가중치 + 편향을 갱신 어떻게? 손실함수 계산을 통해서 => 경사하강법
# 경사하강법 

# ========================================================================= 
# 학습층이 많으면 많을수록 복잡한 문제를 더 잘 해결할꺼라고 생각했지만
# ==> 학습성능이 저하되는 현상이 발생
# ==> 깊이가 깊어 질수록 학습 성능이 저하되는 현상
# ==> 많은 퍼셉트론을 학습하기 위한 컴퓨팅 파워의 한계
# ==> 방대한 데이터의 대한 수집 방법이 부족
# =========================================================================
# ==> 미니 배치 경사하강법, 최적화된 경사하강법
# 
# =========================================================================
# 딥러닝의 대표적인 기본 모델 
#  DNN (Deep Neural Network) => 다수의 은닉층으로 구성된 기본적인 신경망
#  CNN (Convoultional Neural Network) => 영상 처리에 활용되는 합성곱을 이용한 신경망
#  RNN (Recurrent Neural Network) => 계층의 출력이 순환하며 시계열 정보에서 사용되는 신경망
#  Auto Encoder => 입력 데이터의 특징을 추출하는 신경망
# =========================================================================
#

# => Step Function 
# 뉴런을 모방해서 특정 값 이상이 들어왔을때 1 그게 아니면 0 
# 퍼셉트론에서 나오는 값들에 대해서 활성화 시켜주는것 => 활성화 함수    
# Step Function 문제점 => 비선형 문제는 연산이 불가능하다.
# 다층 퍼셉트론 => 
# if => 활성화 함수가 없는 다층 퍼셉트론 
# ㅁ w1*x => ㅁ w1*w2*x=> ㅁ  w1*w2*w3*x=> 0 (w1*w2*w3)*x
# 결국 선형 구조  =>  활성화 함수가 없다면 다층신경망은 비선형 무제의 해결이 불가능하다.
# 활성함수로 모델의 복잡도를 높여 비선형 문제를 해결한다.

# 역전파 알고리즘 =============================================================
#       
#   순전파 => 입력층 부터 출력층까지 순서대로 변수들을 계산
#   비용함수 계산 => 출력층의 결과와 실제값을 활용하여 손실함수 계산
#   MSE 를 주로쓴다.  => 역전파
#   역전파 => 출력층에서 입력층으로 역행하여 가중치와 편향을 갱신
#  


# 합성곱층에서 이미지를 처리하여 다양한 저수준 특징을 추출한다.
# 계단 함수 => 특정구간에서 미분 불가능 => 
# 시그모이드 => 0 보다 작을때는 0.5보다 작은값을 주고 0보다 클때는 0.5보다 큰값을 배출
# 
# 풀링층에서

# CNN 모델 정의
class BirdCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 합성곱층
        # Conv2d : 2D 합성곱 연산을 수행
        # Padding : 이미지의 크기를 유지하기 위해 입력 데이터에 0 을 추가
        # Stride: 필터가 이미지를 스캔할때 한번에 이동하는 간격
        
        # 특징 추출 부분 - Sequential 사용
        # 
        # in_channels : 입력 채널수 (ex: RGB 이미지는 3체널)
        # out_channels : 출력 채널수, 즉 필터의 개수
        # padding : 출력크기가 입력 크기와 동일하게 만들기 위해 패딩을 적용
        
        self.features = nn.Sequential(
            # 첫 번째 합성곱 블록
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        #    nn.AvgPool2d(2,2),
        # 문제
        # ============================================== 
        # 합성곱층을 여러번 쌓으면 이미지의 점차적인 특징 추출이 가능하다.
        # 그렇다면 많이 많이 쌓으면?
        # 

            # 두 번째 합성곱 블록
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # 왜 ? MaxPooling 을 사용하는가? 
            # MaxPooling : 특정 영역 내에서 최대 값을 선택
            # AveragePooling : 특정 영역 내에서 평균값을 선택
    
            nn.MaxPool2d(2, 2),
       #     nn.AvgPool2d(2,2),
            # # 세번째
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        #     # 왜 ? MaxPooling 을 사용하는가? 
        #     # MaxPooling : 특정 영역 내에서 최대 값을 선택
        #     # AveragePooling : 특정 영역 내에서 평균값을 선택
    
            nn.MaxPool2d(2, 2),
             #  nn.AvgPool2d(2,2),
             # 네번째
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
       #     nn.ReLU(),
        #     # 왜 ? MaxPooling 을 사용하는가? 
        #     # MaxPooling : 특정 영역 내에서 최대 값을 선택
        #     # AveragePooling : 특정 영역 내에서 평균값을 선택
           # nn.AvgPool2d(2,2),
            nn.MaxPool2d(2, 2),
        #     # 다섯번째
            nn.Conv2d(32,  32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
    
            nn.MaxPool2d(2, 2),

            nn.Flatten()
        )
        
        # 크기 자동 계산
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)
            dummy_output = self.features(dummy_input)
            flattened_size = dummy_output.shape[1]
            print(f"자동 계산된 특징 크기: {flattened_size}")
        
        # 분류 부분 - Sequential 사용
        self.classifier = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # 특징 추출
        features = self.features(x)
        
        # 분류
        out = self.classifier(features)
        
        return out
    
# ======================================================================================================
#   
#           학습파트
#   
# ======================================================================================================



# 학습 파라미터 설정
num_epochs = 20
batch_size = 12
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BirdCNN().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 모델 파라미터 수 계산
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

total_params = count_parameters(model)

# 데이터 로더 생성
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# 학습 파라미터 출력
print("\n" + "="*50)
print("학습 파라미터 정보:")
print("="*50)
print(f"에폭 수: {num_epochs}")
print(f"배치 크기: {batch_size}")
print(f"학습률: {learning_rate}")
print(f"옵티마이저: Adam")
print(f"손실 함수: CrossEntropyLoss")
print(f"훈련 데이터 크기: {len(train_dataset)}")
print(f"테스트 데이터 크기: {len(test_dataset)}")
print(f"모델 파라미터 수: {total_params:,}")
print(f"사용 장치: {device}")
print("="*50 + "\n")

# 학습 함수
def train_epoch(model, data_loader, loss_fn, optimizer, device, epoch_num):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    epoch_start_time = time.time()
    
    for batch_idx, (inputs, labels) in enumerate(data_loader):
        batch_start_time = time.time()
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        batch_end_time = time.time()
        batch_time = batch_end_time - batch_start_time
        
        ''' 너무 많은 프린트 싫으면 이 부분 주석 처리'''
        # 진행 상황 출력 (10배치마다)
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(data_loader):
            print(f'에폭 {epoch_num+1}/{num_epochs} | 배치 {batch_idx+1}/{len(data_loader)} | '
                  f'손실: {loss.item():.4f} | 배치 처리 시간: {batch_time:.2f}초')
    
    epoch_end_time = time.time()
    epoch_time = epoch_end_time - epoch_start_time
    
    return running_loss / len(data_loader), 100. * correct / total, epoch_time

# 평가 함수
def evaluate(model, data_loader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    eval_start_time = time.time()
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    eval_end_time = time.time()
    eval_time = eval_end_time - eval_start_time
    
    return running_loss / len(data_loader), 100. * correct / total, eval_time

# 시간 형식화 함수
def format_time(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))

# 학습 실행
best_acc = 0.0
best_model_path = 'best_bird_model.pth'

train_losses, train_accs = [], []
test_losses, test_accs = [], []

total_start_time = time.time()

print("학습 시작...")
for epoch in range(num_epochs):
    epoch_start = time.time()
    train_loss, train_acc, train_time = train_epoch(model, train_loader, loss_fn, optimizer, device, epoch)
    test_loss, test_acc, test_time = evaluate(model, test_loader, loss_fn, device)
    
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    
    print("\n" + "-"*70)
    print(f'에폭 {epoch+1}/{num_epochs} 결과:')
    print(f'훈련 손실: {train_loss:.4f}, 훈련 정확도: {train_acc:.2f}%')
    print(f'테스트 손실: {test_loss:.4f}, 테스트 정확도: {test_acc:.2f}%')
    print(f'훈련 시간: {format_time(train_time)} ({train_time:.2f}초)')
    print(f'평가 시간: {format_time(test_time)} ({test_time:.2f}초)')
    
    # 경과 시간 계산 및 출력
    elapsed_time = time.time() - total_start_time
    estimated_total = (elapsed_time / (epoch + 1)) * num_epochs
    remaining_time = estimated_total - elapsed_time
    
    print(f'총 경과 시간: {format_time(elapsed_time)} ({elapsed_time:.2f}초)')
    print(f'예상 남은 시간: {format_time(remaining_time)} ({remaining_time:.2f}초)')
    
    # 현재 모델이 이전에 저장된 최고 모델보다 성능이 좋으면 저장
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), best_model_path)
        print(f'새로운 최고 성능 모델이 저장되었습니다! 정확도: {best_acc:.2f}%')
    print("-"*70 + "\n")

total_end_time = time.time()
total_training_time = total_end_time - total_start_time

print(f'학습 완료!')
print(f'총 학습 시간: {format_time(total_training_time)} ({total_training_time:.2f}초)')
print(f'최종 최고 성능 모델 정확도: {best_acc:.2f}%')

# 결과 시각화
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curves')

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Accuracy')
plt.plot(test_accs, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Accuracy Curves')

plt.tight_layout()
plt.savefig('training_results.png')  # 결과 저장
plt.show()

# 훈련 결과 요약 출력
print("\n" + "="*50)
print("훈련 결과 요약:")
print("="*50)
print(f"에폭 수: {num_epochs}")
print(f"배치 크기: {batch_size}")
print(f"학습률: {learning_rate}")
print(f"최종 훈련 손실: {train_losses[-1]:.4f}")
print(f"최종 훈련 정확도: {train_accs[-1]:.2f}%")
print(f"최종 테스트 손실: {test_losses[-1]:.4f}")
print(f"최종 테스트 정확도: {test_accs[-1]:.2f}%")
print(f"최고 테스트 정확도: {best_acc:.2f}%")
print(f"총 훈련 시간: {format_time(total_training_time)}")
print(f"평균 에폭 시간: {format_time(total_training_time/num_epochs)}")
print("="*50)