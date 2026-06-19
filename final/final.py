import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from facenet_pytorch import MTCNN
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm

# Qiskit 관련 라이브러리 (양자 컴퓨팅 시뮬레이션 및 최적화용)
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer

import warnings
warnings.filterwarnings("ignore")

# ==========================================
# 실험 환경 설정 및 하이퍼파라미터
# ==========================================
# 실행 스크립트와 동일한 위치에 있는 archive 폴더를 참조하도록 상대 경로 지정
BASE_DIR = os.path.join("archive", "dataset_processed_split")
CSV_PATH = os.path.join(BASE_DIR, "dataset_manifest.csv")

# 알고리즘 주요 파라미터
NUM_BLOCKS_K = 32           # 특징(Feature)을 분할할 블록의 개수
TARGET_FEATURES_TOTAL = 256 # 최종적으로 선택할 특징의 총 개수
LAPLACIAN_THRESHOLD = 100.0 # 이미지 흐림(Blur) 정도를 판별하는 기준값 (낮을수록 더 엄격함)

# GPU 사용 가능 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 1. 데이터셋 구성 및 전처리 과정
# ==========================================
class DeepfakeDataset(Dataset):
    def __init__(self, csv_file, base_dir, split='train', transform=None):
        self.base_dir = base_dir
        
        # 메타데이터 로드 및 데이터 분할(train/test) 적용
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        self.transform = transform
        
        # MTCNN: 이미지에서 얼굴만 찾아내기 위한 모델
        # keep_all=False, select_largest=True를 주어 프레임 내에서 가장 크고 뚜렷한 얼굴 하나만 가져옴
        self.mtcnn = MTCNN(keep_all=False, select_largest=True, device=device)
        
    def __len__(self):
        return len(self.df)

    def is_blurry(self, image):
        # 라플라시안 분산(Laplacian Variance)을 이용해 이미지가 흐릿한지 판별함
        # 값이 threshold보다 작으면 윤곽선이 뚜렷하지 않은 저품질 이미지로 간주함
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var() < LAPLACIAN_THRESHOLD

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = 1 if row['label'] == 'FAKE' else 0
        img_path = os.path.join(self.base_dir, row['split'], row['fake_type'], row['filename'])
        
        try:
            image = cv2.imread(img_path)
            if image is None: 
                raise ValueError("Image not found")
            
            # OpenCV는 BGR로 읽어오므로 RGB로 변환 필요
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 학습 데이터일 경우, 너무 흐릿한 이미지는 학습에 방해되므로 더미값과 -1(무시용 라벨)을 반환하여 스킵 유도
            if self.is_blurry(image) and row['split'] == 'train':
                return torch.zeros((3, 160, 160)), -1 

            # MTCNN으로 얼굴 영역만 크롭
            face = self.mtcnn(image)
            
            if face is None:
                # 가끔 MTCNN이 얼굴을 못 찾는 경우가 있음. 이때는 원본 이미지를 그냥 리사이즈해서 사용
                face = cv2.resize(image, (160, 160))
                # HWC 형태를 CHW 형태로 바꾸고 0~1 사이로 정규화
                face = torch.tensor(face).permute(2, 0, 1).float() / 255.0
                
            if self.transform:
                face = self.transform(face)
                
            return face, label
            
        except Exception:
            # 파일이 깨졌거나 읽기 실패한 경우 예외 처리
            return torch.zeros((3, 160, 160)), -1

# ==========================================
# 2. Backbone 특징 추출 네트워크
# ==========================================
def get_backbone():
    # 사전 학습된 ResNet18을 가져와서 분류기(마지막 FC 레이어) 직전까지만 잘라냄
    # 이렇게 하면 이미지의 임베딩 벡터(특징)만 추출할 수 있음
    weights = models.ResNet18_Weights.DEFAULT
    resnet = models.resnet18(weights=weights)
    modules = list(resnet.children())[:-1]
    
    backbone = nn.Sequential(*modules)
    backbone.eval() # 특징 추출만 할 것이므로 평가 모드로 설정
    return backbone.to(device)

def extract_features(dataloader, model, desc="특징 추출 진행"):
    features, labels = [], []
    with torch.no_grad(): # 역전파 비활성화 (메모리 절약 및 속도 향상)
        for x, y in tqdm(dataloader, desc=desc, ncols=100):
            x = x.to(device)
            
            # 앞서 Dataset에서 라벨을 -1로 보낸(흐릿한 이미지 등) 데이터는 걸러냄
            valid_idx = y != -1
            if not valid_idx.any(): 
                continue
            
            x, y = x[valid_idx], y[valid_idx]
            
            # 특징 추출 및 불필요한 차원(1x1 공간 차원) 제거
            out = model(x).squeeze()
            # 배치 사이즈가 1인 경우 차원이 풀려버릴 수 있으므로 복구해줌
            if out.dim() == 1: 
                out = out.unsqueeze(0)
            
            features.append(out.cpu().numpy())
            labels.append(y.numpy())
            
    if not features: 
        return np.array([]), np.array([])
        
    return np.vstack(features), np.concatenate(labels)

# ==========================================
# 3. M-FIG(Multi-Feature Interaction Graph) 구축
# ==========================================
def build_mfig_and_cluster(X, y, num_blocks=32):
    # X: 추출된 특징 벡터 (데이터 수 x 차원 수)
    N, D = X.shape
    
    print("  -> 상호 정보량(Mutual Information) 기반 활성화 벡터 판별력 계산 중...")
    # 각 특징이 정답(FAKE/REAL)을 맞추는 데 얼마나 유용한지 점수를 매김
    mi_scores = mutual_info_classif(X, y)
    
    print("  -> M-FIG 구축: 전체 데이터셋 기준 활성화 패턴 유클리드 거리 연산 중...")
    X_T = X.T 
    # 특징들 간의 거리 행렬을 계산함. 거리가 가까울수록 비슷한 성질을 가진 특징임
    dist_matrix = euclidean_distances(X_T, X_T) 
    
    # 블록 내부 간선 가중치 정의: 절댓값 피어슨 상관계수
    # 특징들 간의 상관관계를 구함. 나중에 중복되는 특징을 제거할 때 사용함
    corr_matrix = np.abs(np.corrcoef(X_T))
    
    print("  -> 그래프 클러스터링: 상호 관련성에 따른 독립적 최적화 블록 분할 중...")
    # 거리를 기반으로 비슷한 특징들끼리 묶어 블록(클러스터)을 만듦
    # 한 번에 양자 최적화를 돌리면 너무 무거우므로 쪼개서 처리하기 위함임
    clustering = AgglomerativeClustering(n_clusters=num_blocks, metric='precomputed', linkage='average')
    block_labels = clustering.fit_predict(dist_matrix)
    
    # 각 블록에 속한 특징들의 인덱스를 리스트 형태로 정리
    initial_blocks = [np.where(block_labels == k)[0] for k in range(num_blocks)]
    
    return initial_blocks, mi_scores, corr_matrix

# ==========================================
# 4. 블록 단위 QAOA 기반 특징 선택
# ==========================================
def select_features_qaoa(initial_blocks, mi_scores, corr_matrix, target_features=256):
    num_blocks = len(initial_blocks)
    total_features = len(mi_scores)
    selected_feature_candidates = []
    
    # 양자 시뮬레이션 환경 구성 (Layer depth p=3, COBYLA 200 iter)
    # QAOA(Quantum Approximate Optimization Algorithm) 설정
    sampler = StatevectorSampler()
    qaoa = QAOA(sampler=sampler, optimizer=COBYLA(maxiter=200), reps=3)
    optimizer = MinimumEigenOptimizer(qaoa)
    
    for k, indices_in_block in enumerate(tqdm(initial_blocks, desc="[블록 단위 QAOA 최적화]", ncols=100)):
        block_size = len(indices_in_block)
        if block_size == 0: 
            continue
            
        # 양자 시뮬레이터 한계 상 큐비트 수가 너무 많으면 메모리가 터질 수 있음(OOM 방지)
        # 한 블록당 최대 15개까지만 허용하고, 넘치면 MI 점수가 높은 순으로 미리 자름
        MAX_QUBITS = 15
        if block_size > MAX_QUBITS:
            top_idx = np.argsort(mi_scores[indices_in_block])[-MAX_QUBITS:]
            indices_in_block = indices_in_block[top_idx]
            block_size = MAX_QUBITS

        # 해당 블록에서 최종적으로 몇 개의 특징을 뽑을지 결정 (블록의 중요도에 비례해서 동적 할당)
        block_mi_sum = np.sum(mi_scores[indices_in_block])
        total_mi_sum = np.sum(mi_scores)
        nk = max(1, int(target_features * (block_mi_sum / total_mi_sum)))
        nk = min(nk, block_size)

        # QUBO(Quadratic Unconstrained Binary Optimization) 모델 설계
        # lam(상관관계 페널티 가중치), mu(선택 개수 제약 페널티 가중치)
        lam, mu = 0.3, 10.0
        qp = QuadraticProgram()
        
        # 블록 내 각 특징을 선택할지 말지 결정하는 이진 변수(0 또는 1) 생성
        for i in range(block_size): 
            qp.binary_var(f'z_{i}')
            
        linear_terms, quadratic_terms = {}, {}
        
        # 목적 함수 구성
        for i in range(block_size):
            orig_i = indices_in_block[i]
            # 1차항: MI 점수가 높을수록 좋으므로 음수(-) 처리, 개수 맞추기 위한 페널티 추가
            linear_terms[f'z_{i}'] = -mi_scores[orig_i] + mu * (1 - 2 * nk)
            
            # 2차항: 서로 비슷한 특징을 동시에 뽑으면 페널티 부여(중복성 최소화)
            for j in range(i + 1, block_size):
                orig_j = indices_in_block[j]
                quad_val = lam * corr_matrix[orig_i, orig_j] + 2 * mu
                quadratic_terms[(f'z_{i}', f'z_{j}')] = quad_val
                
        # 설계한 수식들을 최적화 모델에 반영
        qp.minimize(linear=linear_terms, quadratic=quadratic_terms, constant=mu * (nk ** 2))
        
        try:
            # QAOA 알고리즘으로 해를 탐색
            result = optimizer.solve(qp)
            # 결과가 1.0(선택됨)인 인덱스만 추출
            selected_in_block = [indices_in_block[i] for i, bit in enumerate(result.x) if bit == 1.0]
            selected_feature_candidates.extend(selected_in_block)
        except Exception:
            # 행렬 조건이 안 맞거나 시뮬레이션 에러가 날 경우, 안전하게 고전적 방식(Greedy)으로 넘어감
            top_k_idx = np.argsort(mi_scores[indices_in_block])[-nk:]
            selected_feature_candidates.extend(indices_in_block[top_k_idx])

    # 유연한 컷오프(Soft Cutoff)
    # 각 블록에서 뽑아온 특징들의 총합이 우리가 원하는 개수(target_features)와 다를 수 있으므로 맞춰줌
    candidate_indices = list(set(selected_feature_candidates))
    
    if len(candidate_indices) > target_features:
        # 너무 많이 뽑혔으면 MI 점수 낮은 애들부터 버림
        candidate_indices = sorted(candidate_indices, key=lambda idx: mi_scores[idx], reverse=True)[:target_features]
    elif len(candidate_indices) < target_features:
        # 모자라면 안 뽑힌 애들 중에서 MI 점수 높은 애들로 채워 넣음
        remaining_indices = [idx for idx in range(total_features) if idx not in candidate_indices]
        remaining_indices = sorted(remaining_indices, key=lambda idx: mi_scores[idx], reverse=True)
        needed = target_features - len(candidate_indices)
        candidate_indices.extend(remaining_indices[:needed])
        
    return np.array(candidate_indices)

# ==========================================
# 5. 양자 확률 기반 어텐션 분류기
# ==========================================
class QuantumHybridClassifier(nn.Module):
    def __init__(self, input_dim):
        super(QuantumHybridClassifier, self).__init__()
        
        # 본의 규칙(Born's rule)에 착안한 확률 진폭 파라미터
        # 이 값을 학습시켜 입력 특징들에 곱해줄 가중치를 결정함
        self.theta = nn.Parameter(torch.randn(input_dim))
        
        # 추출된 특징 간의 관계를 파악하기 위한 멀티 헤드 어텐션
        self.mha = nn.MultiheadAttention(embed_dim=input_dim, num_heads=8, batch_first=True)
        
        # 최종적으로 FAKE/REAL을 가려낼 분류기 구조
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64), 
            nn.ReLU(), 
            nn.Dropout(0.3),  # 과적합 방지
            nn.Linear(64, 1), 
            nn.Sigmoid()      # 0~1 사이의 확률값으로 출력
        )
        
    def forward(self, x):
        # 파라미터를 제곱 단위(sin^2)로 바꿔서 0~1 사이의 확률값처럼 다룸
        born_probs = torch.sin(self.theta) ** 2 
        
        # 입력값에 확률 가중치를 곱해 중요도를 재조정 (일종의 소프트 어텐션)
        x_attn1 = x * born_probs
        
        # MultiheadAttention은 시퀀스 형태의 입력을 원하므로 차원을 잠시 늘려줌 (N, 1, D)
        x_seq = x_attn1.unsqueeze(1) 
        attn_output, _ = self.mha(x_seq, x_seq, x_seq)
        
        # 잔차 연결(Residual Connection) 사용
        x_out = x_attn1 + attn_output.squeeze(1)
        
        return self.fc(x_out)

# ==========================================
# 실험 실행 파이프라인
# ==========================================
def main():
    print("\n" + "="*80)
    print("[연구 실험] M-FIG 기반 하이브리드 양자 최적화 딥페이크 탐지 파이프라인")
    print("="*80)
    
    print("\n[1 & 2] 데이터셋 로드 및 고차원 특징(Feature) 추출")
    print("-" * 80)
    
    # ImageNet에서 쓰이는 기본 정규화 수치 적용
    transform = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    train_dataset = DeepfakeDataset(CSV_PATH, BASE_DIR, split='train', transform=transform)
    test_dataset = DeepfakeDataset(CSV_PATH, BASE_DIR, split='test', transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    backbone = get_backbone()
    
    # ResNet을 돌려서 원본 이미지에서 512차원 벡터들을 뽑아냄
    X_train, y_train = extract_features(train_loader, backbone, desc="[학습 데이터] ResNet 특징 추출")
    X_test, y_test = extract_features(test_loader, backbone, desc="[검증 데이터] ResNet 특징 추출")
    
    if len(X_train) == 0:
        print("\n오류: 특징 추출 실패. 데이터 구조 및 경로 확인 필요.")
        return
        
    print(f"\n특징 추출 완료. 초기 행렬 차원: {X_train.shape}")
    
    print("\n" + "="*80)
    print(f"[3] M-FIG 기반 특징 공간 구조화 (설정된 블록 수: {NUM_BLOCKS_K})")
    print("-" * 80)
    
    initial_blocks, mi_scores, corr_matrix = build_mfig_and_cluster(X_train, y_train, num_blocks=NUM_BLOCKS_K)
    print(f"\n특징 공간 분할 완료. 전체 512차원 특징이 {len(initial_blocks)}개의 블록으로 구조화됨.")
    
    print("\n" + "="*80)
    print("[4] 블록 단위 양자 근사 최적화(Block-QAOA) 및 최종 특징 선별")
    print("-" * 80)
    
    # QAOA를 활용해 중복되는 특징은 쳐내고 판별력 높은 알짜배기 특징만 추려냄
    selected_indices = select_features_qaoa(initial_blocks, mi_scores, corr_matrix, target_features=TARGET_FEATURES_TOTAL)
    final_dim = len(selected_indices)
    print(f"\n양자 최적화 및 유연한 컷오프(Soft Cutoff) 적용 완료. 최종 사용 차원: {final_dim} 차원")
    
    if final_dim == 0:
        print("\n오류: 유효한 특징이 선별되지 않음.")
        return

    # 원본 특징들 중에서 선택된 특징 인덱스만 남김
    X_train_q, X_test_q = X_train[:, selected_indices], X_test[:, selected_indices]

    print("\n" + "="*80)
    print("[5] 양자 확률 기반 어텐션 분류기 학습 및 성능 평가")
    print("-" * 80)
    
    model = QuantumHybridClassifier(input_dim=final_dim).to(device)
    criterion = nn.BCELoss() # 이진 분류이므로 Binary Cross Entropy 사용
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Scikit-learn 배열을 PyTorch 텐서로 변환
    X_train_t = torch.tensor(X_train_q, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test_q, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    
    epochs = 5
    for epoch in tqdm(range(epochs), desc="[모델 학습 진행도]", ncols=100):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(X_train_t.to(device))
        loss = criterion(outputs, y_train_t.to(device))
        loss.backward()
        optimizer.step()
        
    # 평가 모드 전환 및 성능 지표 계산
    model.eval()
    with torch.no_grad():
        preds_probs = model(X_test_t.to(device)).cpu().numpy()
        preds = (preds_probs > 0.5).astype(int)
        
        acc = accuracy_score(y_test_t.numpy(), preds)
        prec = precision_score(y_test_t.numpy(), preds, zero_division=0)
        rec = recall_score(y_test_t.numpy(), preds, zero_division=0)
        f1 = f1_score(y_test_t.numpy(), preds, zero_division=0)
        
        try: 
            auc = roc_auc_score(y_test_t.numpy(), preds_probs)
        except ValueError: 
            auc = 0.0
            
        print("\n" + "="*80)
        print("[최종 평가 결과] 제안 방법론 성능 지표")
        print("="*80)
        print(f"  * 정확도   (Accuracy)  : {acc:.4f}")
        print(f"  * 정밀도   (Precision) : {prec:.4f}")
        print(f"  * 재현율   (Recall)    : {rec:.4f}")
        print(f"  * F1-점수  (F1-Score)  : {f1:.4f}")
        print(f"  * AUC-ROC             : {auc:.4f}")
        print("="*80)
        print("모든 실험 절차 종료.\n")

if __name__ == "__main__":
    main()