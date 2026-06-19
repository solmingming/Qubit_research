# M-FIG and Block-QAOA based Hybrid Deepfake Detection

This repository contains the final implementation code for a deepfake detection experiment using a hybrid pipeline that combines classical deep learning-based feature extraction with quantum-inspired optimization.

The main purpose of this project is to detect deepfake images by extracting high-dimensional visual features from face images, structuring the feature space using M-FIG, selecting important features using Block-QAOA, and training a quantum-inspired attention classifier for final binary classification.

---

## 1. Project Overview

Deepfake detection is a binary classification task that determines whether a given face image is real or fake.

In this project, we propose a hybrid pipeline consisting of the following steps:

1. Face detection and preprocessing using MTCNN
2. Feature extraction using pretrained ResNet18
3. Feature interaction graph construction using M-FIG
4. Feature block clustering using agglomerative clustering
5. Block-wise feature selection using QAOA
6. Binary classification using a quantum-inspired attention classifier

The final output of the program is a set of classification metrics, including accuracy, precision, recall, F1-score, and AUC-ROC.

---

## 2. Repository Structure

The current repository is organized as follows:

```bash
final/
├── archive/
│   └── dataset_processed_split/
│       ├── dataset_manifest.csv
│       ├── train/
│       │   ├── Real/
│       │   ├── DeepFakeDetection/
│       │   ├── Deepfakes/
│       │   ├── Face2Face/
│       │   ├── FaceShifter/
│       │   ├── FaceSwap/
│       │   └── NeuralTextures/
│       └── test/
│           ├── Real/
│           ├── DeepFakeDetection/
│           ├── Deepfakes/
│           ├── Face2Face/
│           ├── FaceShifter/
│           ├── FaceSwap/
│           └── NeuralTextures/
│
├── final.py
├── requirements.txt
└── README.md
```

The main executable file is:

```bash
final.py
```

The dataset is expected to be located at:

```bash
archive/dataset_processed_split/
```

The metadata file is expected to be located at:

```bash
archive/dataset_processed_split/dataset_manifest.csv
```

---

## 3. Dataset Format

The code loads dataset information from:

```bash
archive/dataset_processed_split/dataset_manifest.csv
```

The CSV file should contain at least the following columns:

| Column      | Description                                                   |
| ----------- | ------------------------------------------------------------- |
| `split`     | Dataset split. Expected values: `train` or `test`             |
| `label`     | Ground-truth label. Expected values: `REAL` or `FAKE`         |
| `fake_type` | Subfolder name such as `Real`, `Deepfakes`, `Face2Face`, etc. |
| `filename`  | Image file name                                               |

The image path is automatically constructed in the following format:

```bash
archive/dataset_processed_split/{split}/{fake_type}/{filename}
```

For example:

```bash
archive/dataset_processed_split/train/Real/example.jpg
archive/dataset_processed_split/test/Deepfakes/example.jpg
```

---

## 4. Environment Setup

This project is recommended to run with Python 3.9 or 3.10.

### 4.1 Clone the Repository

```bash
git clone https://github.com/solmingming/Qubit_research.git
cd Qubit_research/final
```

It is important to move into the `final` directory before running the code because the dataset path in `final.py` is defined relative to the `final` directory.

---

### 4.2 Create a Virtual Environment

Using `venv`:

```bash
python -m venv venv
source venv/bin/activate
```

For Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Using `conda`:

```bash
conda create -n qubit-final python=3.10
conda activate qubit-final
```

---

### 4.3 Install Dependencies

Install all required packages using:

```bash
pip install -r requirements.txt
```

If PyTorch installation fails, install PyTorch separately according to your local CPU/GPU environment, and then install the remaining packages again.

---

## 5. How to Run

After cloning the repository and installing dependencies, run:

```bash
python final.py
```

The program will automatically execute the full experimental pipeline.

---

## 6. Execution Pipeline

When `final.py` is executed, the following steps are performed.

### Step 1. Dataset Loading and Face Preprocessing

The code loads the dataset based on `dataset_manifest.csv`.

For each image:

1. The image is loaded with OpenCV.
2. The image is converted from BGR to RGB.
3. Blurry training images are filtered using Laplacian variance.
4. MTCNN detects and crops the main face region.
5. If face detection fails, the original image is resized and used instead.

---

### Step 2. Feature Extraction with ResNet18

A pretrained ResNet18 model is used as a feature extractor.

The final classification layer of ResNet18 is removed, and each image is converted into a 512-dimensional feature vector.

Output example:

```text
특징 추출 완료. 초기 행렬 차원: (N, 512)
```

Here, `N` is the number of valid images after preprocessing.

---

### Step 3. M-FIG Construction

The extracted feature vectors are used to construct an M-FIG, or Multi-Feature Interaction Graph.

This step calculates:

1. Mutual information score between each feature and the label
2. Euclidean distance between feature activation patterns
3. Pearson correlation between features

These values are used to evaluate feature importance and feature redundancy.

---

### Step 4. Feature Block Clustering

The 512-dimensional feature space is divided into several blocks using agglomerative clustering.

The number of blocks is controlled by:

```python
NUM_BLOCKS_K = 32
```

The purpose of this step is to reduce the complexity of QAOA optimization by applying feature selection block by block.

---

### Step 5. Block-QAOA Feature Selection

For each feature block, QAOA is used to select important and less redundant features.

The code formulates a QUBO problem using:

1. Mutual information score as feature importance
2. Correlation score as redundancy penalty
3. Selection-count penalty to control the number of selected features

The target number of final selected features is controlled by:

```python
TARGET_FEATURES_TOTAL = 256
```

If QAOA optimization fails for a specific block, the code automatically falls back to a greedy mutual-information-based selection method.

---

### Step 6. Quantum-inspired Attention Classifier

After feature selection, the selected features are passed into `QuantumHybridClassifier`.

The classifier consists of:

1. Born's rule-inspired trainable probability weights
2. Multi-head attention layer
3. Fully connected binary classifier

The classifier outputs a probability between 0 and 1.

```text
0 → REAL
1 → FAKE
```

---

### Step 7. Final Evaluation

The model is evaluated on the test dataset using the following metrics:

| Metric    | Description                                                            |
| --------- | ---------------------------------------------------------------------- |
| Accuracy  | Overall classification correctness                                     |
| Precision | Ratio of correctly predicted fake samples among predicted fake samples |
| Recall    | Ratio of correctly detected fake samples among actual fake samples     |
| F1-score  | Harmonic mean of precision and recall                                  |
| AUC-ROC   | Area under the ROC curve                                               |

Example output:

```text
================================================================================
[최종 평가 결과] 제안 방법론 성능 지표
================================================================================
  * 정확도   (Accuracy)  : 0.XXXX
  * 정밀도   (Precision) : 0.XXXX
  * 재현율   (Recall)    : 0.XXXX
  * F1-점수  (F1-Score)  : 0.XXXX
  * AUC-ROC             : 0.XXXX
================================================================================
모든 실험 절차 종료.
```

---

## 7. Main Hyperparameters

The main hyperparameters are defined at the top of `final.py`.

| Variable                | Default Value | Description                                            |
| ----------------------- | ------------: | ------------------------------------------------------ |
| `NUM_BLOCKS_K`          |          `32` | Number of feature blocks                               |
| `TARGET_FEATURES_TOTAL` |         `256` | Final number of selected features                      |
| `LAPLACIAN_THRESHOLD`   |       `100.0` | Threshold for filtering blurry images                  |
| `MAX_QUBITS`            |          `15` | Maximum number of features optimized in one QAOA block |
| `epochs`                |           `5` | Number of classifier training epochs                   |
| `batch_size`            |          `32` | DataLoader batch size                                  |
| `learning_rate`         |       `0.001` | Adam optimizer learning rate                           |

---

## 8. Reproducing the Experiment

To reproduce the experiment from the current repository state, run the following commands:

```bash
git clone https://github.com/solmingming/Qubit_research.git
cd Qubit_research/final

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
python final.py
```

For Windows:

```bash
git clone https://github.com/solmingming/Qubit_research.git
cd Qubit_research/final

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
python final.py
```

---

## 9. Important Notes

### 9.1 Run the Code inside the `final` Directory

The code uses the following relative path:

```python
BASE_DIR = os.path.join("archive", "dataset_processed_split")
CSV_PATH = os.path.join(BASE_DIR, "dataset_manifest.csv")
```

Therefore, the recommended execution location is:

```bash
Qubit_research/final
```

If the code is executed from the repository root, the dataset path may not be found.

---

### 9.2 Dataset Split Requirement

The code uses both train and test splits:

```python
train_dataset = DeepfakeDataset(CSV_PATH, BASE_DIR, split='train', transform=transform)
test_dataset = DeepfakeDataset(CSV_PATH, BASE_DIR, split='test', transform=transform)
```

Therefore, `dataset_manifest.csv` should contain both `train` and `test` rows, and the corresponding image files should exist under the matching folders.

If the train split is missing, the model cannot be trained properly.

---

### 9.3 First Run May Require Internet Access

The code uses pretrained ResNet18 weights from `torchvision`.

If the pretrained weights are not already cached on the local machine, they may be downloaded automatically during the first run.

---

### 9.4 CPU and GPU

The code automatically uses GPU if CUDA is available:

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

If CUDA is not available, the code runs on CPU. However, feature extraction and QAOA optimization may take longer.

---

## 10. Troubleshooting

### 10.1 `FileNotFoundError: dataset_manifest.csv`

Check whether the following file exists:

```bash
archive/dataset_processed_split/dataset_manifest.csv
```

Also make sure that you are running the code from the `final` directory:

```bash
cd Qubit_research/final
python final.py
```

---

### 10.2 `ModuleNotFoundError`

If a package is missing, run:

```bash
pip install -r requirements.txt
```

---

### 10.3 OpenCV Import Error

If `cv2` cannot be imported, reinstall OpenCV:

```bash
pip install opencv-python
```

---

### 10.4 PyTorch Installation Error

If PyTorch installation fails, install PyTorch separately according to your device environment.

For CPU-only execution:

```bash
pip install torch torchvision
```

Then run:

```bash
pip install -r requirements.txt
```

---

### 10.5 Qiskit-related Error

If Qiskit-related modules are missing, reinstall the Qiskit packages:

```bash
pip install qiskit qiskit-algorithms qiskit-optimization
```

---

### 10.6 Empty Feature Extraction Result

If the following message appears:

```text
오류: 특징 추출 실패. 데이터 구조 및 경로 확인 필요.
```

Check the following points:

1. `dataset_manifest.csv` contains valid rows.
2. The `split`, `fake_type`, and `filename` columns are correctly written.
3. The actual image files exist in the expected path.
4. The code is being executed from the `final` directory.
5. The train split is not empty.

---

## 11. Expected Result

After successful execution, the terminal should print the final evaluation metrics:

```text
Accuracy
Precision
Recall
F1-Score
AUC-ROC
```

A successful run means that the full pipeline completed without path errors, dependency errors, or dataset loading errors.

---

## 12. Team

Team Qubit

* 이솔민
* 김은솜
* 김정민

Graduation Project
Department of Computer Science and Engineering
Ewha Womans University

---

## 13. License

This repository is intended for academic and educational use.
