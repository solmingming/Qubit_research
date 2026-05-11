# Team-01
| 항목 | 내용 |
|:---|:---|
| (1) 과제명 | Graph-Clustered Block-QAOA Feature Selection for Deepfake Detection |
| (2) 팀 번호/이름 | 01-Qubit |
| (3) 구성원 | 이솔민(2271046): 리더, QUBO 모델링, 양자 최적화 파이프라인 설계 <br> 김정민(2271020): 팀원, 데이터셋 전처리, 평가 지표 구축 및 시각화 <br> 김은솜(2271018): 팀원, 베이스라인 모델 개발, feature 추출, M-FIG 클러스터링 구현 |
| (4) 지도교수 | 이형준 교수 |
| (5) 트랙 | 연구 |
| (6) 과제 키워드 | 딥페이크 탐지, Feature Selection, M-FIG, Graph Clustering, Block-QAOA, QUBO, 조합 최적화 |
| (7) 과제 내용 요약 | **[한줄소개]** <br> 딥페이크 이미지에서 추출된 512차원 feature를 그래프 구조로 모델링하고, M-FIG 기반 그래프 클러스터링과 Block-QAOA를 결합하여 딥페이크 탐지에 중요한 핵심 feature subset을 선택하는 연구를 수행한다. <br><br> **[연구 배경 및 필요성]** <br> 생성형 AI의 발전으로 딥페이크 이미지는 점점 더 정교해지고 있으며, 단순한 시각적 차이만으로는 진위 여부를 구분하기 어려워지고 있다. 이에 따라 딥페이크 탐지 모델은 고차원 feature representation을 활용하지만, 모든 feature가 탐지 성능에 동일하게 기여하는 것은 아니다. 일부 feature는 중복되거나 노이즈를 포함할 수 있으며, 이는 연산 비용 증가와 모델 해석 가능성 저하로 이어질 수 있다. 따라서 탐지 성능을 유지하면서 핵심 feature subset을 선택하는 효율적인 feature selection 방법이 필요하다. <br><br> **[연구 목적]** <br> 본 연구는 512차원 딥페이크 feature를 M-FIG(Multi-Feature Interaction Graph)로 모델링하고, feature 간 관계를 반영한 그래프 기반 클러스터링을 통해 block을 구성한다. 이후 각 block 내부에서 QUBO 기반 Block-QAOA 최적화를 수행하여 딥페이크 탐지에 중요한 feature subset을 선택한다. 이를 통해 단순 차원 축소가 아니라, feature 간 관계를 보존하는 graph-aware quantum feature selection framework를 구축한다. <br><br> **[핵심 연구 내용 및 방법론]** <br> 1. 딥페이크 이미지에서 얼굴 영역을 검출 및 정렬하고, 사전학습된 face embedding model을 활용하여 512차원 feature를 추출한다. <br> 2. 각 feature dimension을 node로 정의하고, feature 간 상관관계와 의존성을 edge weight로 표현하여 M-FIG를 구성한다. <br> 3. M-FIG 기반 graph clustering을 통해 관련 feature들이 같은 block에 포함되도록 구성한다. <br> 4. 각 block의 feature selection 문제를 QUBO 형태로 정식화하고, Block-QAOA를 적용하여 핵심 feature subset을 선택한다. <br> 5. 선택된 feature subset을 classifier에 입력하여 Fake/Real 분류 성능을 평가하고, full feature 방식 및 sequential block 방식과 비교 분석한다. <br><br> **[실험 및 검증 계획]** <br> FaceForensics++ 데이터셋을 활용하여 제안하는 M-FIG 기반 Graph-Clustering Block-QAOA feature selection pipeline의 효과를 검증한다. 주요 평가지표로는 Detection Accuracy, Feature Compression Ratio, Performance Retention, Optimization Efficiency, Block Quality를 사용한다. 이를 통해 feature 수를 줄였을 때 탐지 성능이 얼마나 유지되는지, graph-based block 구성이 sequential block 방식보다 feature 관계를 더 잘 보존하는지 분석한다. <br><br> **[연구의 독창성 및 기대 효과]** <br> 본 연구는 딥페이크 feature를 단순한 벡터가 아니라 feature 간 관계를 가진 weighted graph로 모델링한다. 또한 NISQ 환경의 제약을 고려하여 고차원 feature selection 문제를 block-wise QAOA 구조로 분할한다. 이를 통해 딥페이크 탐지 성능을 유지하면서도 연산 비용을 줄일 수 있는 경량화된 feature selection 방법을 제안하고, 보안 AI 문제에 대한 양자-클래식 하이브리드 최적화의 응용 가능성을 검토한다. |
| (8) 주요 Link | https://github.com/solmingming/Qubit_research |
 
<br>
