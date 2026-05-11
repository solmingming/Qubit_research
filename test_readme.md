# Team-01
| (1) 과제명 |  Graph-Clustered Block-QAOA Feature Selection for Deepfake Detection |
|:---  |---  |
| (2) 팀 번호/이름 | 01-Qubit |
| (3) 구성원 | 이솔민(2271046): 리더,QUBO모델링, 양자 학습 루프 최적화 <br> 김정민(2271020): 팀원, 데이터셋 전처리 파이프라인 구축,평가 지표 구축 및 시각화 <br> 김은솜(2271018): 팀원,베이스라인 모델 개발, 딥페이크 이미지 feature추출,M-FIG 클러스터링 개발 |
| (4) 지도교수 | 이형준 교수 |
| (5) 트랙  | 연구 |
| (6) 과제 키워드 | 딥페이크 탐지, Feature Selection, M-FIG, Graph Clustering, Block-QAOA, QUBO, 조합 최적화 |
|---|---|
| (7) 과제 내용 요약 | [한줄소개] <br> 딥페이크 이미지에서 추출된 고차원 feature를 그래프 구조로 모델링하고, 그래프 기반 클러스터링과 Block-QAOA를 결합하여 딥페이크 탐지에 중요한 핵심 feature subset을 선택하는 양자-클래식 하이브리드 feature selection 연구를 수행한다.
<br><br> [연구 배경 및 필요성]
<br> - **딥페이크의 고도화와 미세 특징 분석의 필요성:** 생성형 AI 기술의 발전으로 딥페이크 이미지는 점점 더 정교해지고 있으며, 단순한 시각적 차이만으로는 진위 여부를 구분하기 어려워지고 있다. 이에 따라 피부 질감의 비자연성, 얼굴 영역 간 불일치, 생성 모델 특유의 artifact와 같은 미세한 위조 단서를 포착할 수 있는 고차원 feature representation의 중요성이 증가하고 있다.
<br> - **고차원 feature의 비효율성:** 딥페이크 탐지 모델은 높은 탐지 성능을 위해 512차원 이상의 deep feature를 활용하는 경우가 많지만, 모든 feature가 탐지 성능에 동일하게 기여하는 것은 아니다. 일부 feature는 중복되거나 노이즈를 포함할 수 있으며, 이는 연산 비용 증가와 모델 해석 가능성 저하로 이어질 수 있다. 
<br> - **Feature selection의 조합 최적화 문제:** 딥페이크 탐지에 중요한 feature subset을 선택하는 문제는 각 feature를 선택할지 말지를 결정하는 조합 최적화 문제로 볼 수 있다. feature 수가 증가할수록 가능한 조합의 수가 기하급수적으로 증가하므로, 효율적인 최적화 전략이 필요하다.
<br> - **QAOA와 NISQ 환경의 제약:** QAOA(Quantum Approximate Optimization Algorithm)는 조합 최적화 문제에 적용 가능한 대표적인 변분 양자 알고리즘이다. 그러나 현재의 NISQ 환경에서는 qubit 수와 circuit depth의 제약으로 인해 512차원 feature 전체를 한 번에 최적화하기 어렵다. 따라서 본 연구는 feature 전체를 작은 block 단위로 나누어 최적화하는 Block-QAOA 구조를 활용한다.
<br> - **그래프 기반 block 구성의 필요성:** 단순히 feature index 순서대로 block을 나누는 방식은 실제로 서로 관련 있는 feature들을 분리할 수 있다. 이에 본 연구는 feature 간 관계를 반영한 M-FIG(Multi-Feature Interaction Graph)를 구성하고, 그래프 기반 클러스터링을 통해 관련 feature들이 같은 block에 배치되도록 설계한다.
<br><br> [연구 목적] <br> - **목적:** 딥페이크 이미지에서 추출된 512차원 feature를 M-FIG로 모델링하고, 그래프 기반 클러스터링을 통해 feature block을 구성한 뒤, 각 block 내부에서 QUBO 기반 Block-QAOA 최적화를 수행하여 탐지 성능에 중요한 feature subset을 선택한다.
<br> - **포지셔닝:** 본 연구는 딥페이크 탐지를 위한 단순 차원 축소가 아니라, feature 간 관계를 보존하면서 핵심 feature 조합을 선택하는 **graph-aware quantum feature selection framework**를 구축한다.
<br><br> [핵심 연구 내용 및 방법론] 
<br> - **고차원 feature 추출:** 딥페이크 이미지에서 얼굴 영역을 검출 및 정렬한 뒤, 사전학습된 face embedding model을 활용하여 512차원 feature representation을 추출한다.
 <br> - **M-FIG 구성:** 추출된 feature의 각 dimension을 node로 정의하고, feature 간 상관관계, 의존성 또는 유사도를 edge weight로 표현하여 M-FIG(Multi-Feature Interaction Graph)를 구성한다. 
 <br> - **그래프 기반 feature clustering:** M-FIG를 기반으로 서로 관련 있는 feature들이 같은 block에 포함되도록 graph-based clustering을 수행한다. 이를 통해 기존 sequential block 방식에서 발생할 수 있는 feature 관계 손실을 줄인다. 
 <br> - **QUBO 정식화:** 각 feature block 내부에서 feature selection 문제를 QUBO(Quadratic Unconstrained Binary Optimization) 형태로 정식화한다. 각 feature는 선택 여부를 나타내는 binary variable로 표현하며, feature 중요도와 feature 간 관계를 목적 함수에 반영한다. 
 <br> - **Block-QAOA 최적화:** 전체 512차원 feature를 한 번에 최적화하는 대신, block 단위로 QAOA를 적용하여 NISQ 환경에서 처리 가능한 크기의 feature selection 문제로 분할한다. 
 <br> - **딥페이크 분류 성능 평가:** 선택된 feature subset을 classifier의 입력으로 사용하여 Fake/Real 분류 성능을 평가하고, full feature 사용 방식, sequential block 방식, random feature selection 방식 등과 비교 분석한다. 
 <br><br> [실험 및 검증 계획] 
 <br> 본 연구는 딥페이크 탐지 연구에서 널리 사용되는 FaceForensics++ 데이터셋을 활용하여, 제안하는 M-FIG 기반 Graph-Clustering Block-QAOA feature selection pipeline의 효과를 검증한다. 
 <br><br> 데이터셋 구성 및 전처리 
 <br> - **딥페이크 이미지 데이터 활용:** FaceForensics++ 데이터셋을 기반으로 real/fake 이미지 또는 프레임을 구성하고, 얼굴 영역 검출 및 정렬 과정을 거쳐 feature extraction을 수행한다. 
 <br> - **512차원 feature representation 추출:** 사전학습된 face embedding model을 활용하여 각 이미지로부터 512차원 feature vector를 추출하고, 이를 M-FIG 구성 및 feature selection의 입력으로 사용한다. 
 <br> - **Block 구성 비교:** 기존 sequential block 방식과 제안하는 graph-based clustering block 방식을 비교하여, feature 간 관계를 반영한 block 구성이 feature selection 성능에 미치는 영향을 분석한다. 
 <br><br> 주요 평가지표 
 <br> 1. **Detection Accuracy:** 선택된 feature subset을 사용했을 때의 딥페이크 탐지 정확도를 측정한다. 
 <br> 2. **Feature Compression Ratio:** 원본 512차원 feature 대비 선택된 feature 수의 비율을 분석한다. 
 <br> 3. **Performance Retention:** feature 수를 줄였을 때 full feature baseline 대비 탐지 성능이 얼마나 유지되는지 평가한다. 
 <br> 4. **Optimization Efficiency:** sequential block 방식 대비 graph-based block 구성 및 Block-QAOA 적용 시의 최적화 효율성을 비교 분석한다. 
 <br> 5. **Block Quality:** M-FIG 기반 clustering이 feature 간 관계를 얼마나 잘 보존하는지 분석한다. 
 <br><br> [연구의 독창성 및 기대 효과] 
 <br> - **M-FIG 기반 feature interaction modeling:** 딥페이크 feature를 단순한 벡터가 아니라 feature 간 관계를 가진 weighted graph로 모델링함으로써, feature selection 과정에서 구조적 정보를 반영한다. <br> - **Graph-aware block construction:** 기존의 단순 sequential partitioning 방식과 달리, feature 간 상호작용을 반영하여 관련 feature들이 같은 block에 배치되도록 구성한다. 
 <br> - **NISQ-aware Block-QAOA pipeline:** 현재 양자컴퓨팅 환경의 qubit 수 및 circuit depth 제약을 고려하여, 고차원 feature selection 문제를 block-wise QAOA 구조로 분할한다. 
 <br> - **효율적인 딥페이크 탐지 모델 구성:** 탐지에 중요한 feature subset만 선택함으로써, 연산 비용을 줄이면서도 탐지 성능을 유지할 수 있는 경량화된 딥페이크 탐지 구조를 제안한다. <br> - **양자 최적화 기반 보안 AI 응용 가능성 탐색:** 딥페이크 탐지라는 실제 보안 문제에 QAOA 기반 조합 최적화 방법을 적용함으로써, 양자-클래식 하이브리드 최적화의 응용 가능성을 실험적으로 검토한다. |

| (8) 주요 Link | https://github.com/solmingming/Qubit_research |
 
<br>
