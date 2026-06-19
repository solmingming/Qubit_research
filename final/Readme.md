# M-FIG & Block-QAOA based Hybrid Deepfake Detection

<p align="center">
  <b>Graduation Project | Team Qubit</b><br>
  Hybrid Deepfake Detection using Classical Deep Learning and Quantum Optimization
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue" />
  <img src="https://img.shields.io/badge/PyTorch-2.2.2-red" />
  <img src="https://img.shields.io/badge/Qiskit-QAOA-purple" />
  <img src="https://img.shields.io/badge/Task-Deepfake%20Detection-green" />
  <img src="https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey" />
</p>

---

## Table of Contents

* [1. Basic Description](#1-basic-description)
* [2. Computational Resource Notice](#2-computational-resource-notice)
* [3. Project Description](#3-project-description)
* [4. Source Code Description](#4-source-code-description)
* [5. Repository Structure](#5-repository-structure)
* [6. How to Build](#6-how-to-build)
* [7. How to Install](#7-how-to-install)

  * [7.1 Clone the Repository](#71-clone-the-repository)
  * [7.2 Create a Virtual Environment](#72-create-a-virtual-environment)
  * [7.3 Install Required Packages](#73-install-required-packages)
  * [7.4 Installation Check](#74-installation-check)
* [8. How to Test](#8-how-to-test)
* [9. Experimental Results](#9-experimental-results)
* [10. Description of Sample Data](#10-description-of-sample-data)
* [11. Database or Data Used](#11-database-or-data-used)
* [12. Description of Used Open Source](#12-description-of-used-open-source)
* [13. Reproducibility Notes](#13-reproducibility-notes)
* [14. Troubleshooting](#14-troubleshooting)
* [15. Team](#15-team)
* [16. License](#16-license)

---

## 1. Basic Description

This repository contains the final source code, installation requirements, sample dataset, and experiment-related files for Team Qubit's graduation project.

The project implements a **hybrid deepfake detection pipeline** that combines:

* classical deep learning-based image feature extraction,
* graph-based feature interaction analysis,
* quantum optimization-based feature selection,
* and neural network-based binary classification.

The final task is to classify a facial image as either:

| Class          | Label  |
| -------------- | ------ |
| Real image     | `REAL` |
| Deepfake image | `FAKE` |

The main executable file is:

```bash
final.py
```

The dataset is stored in:

```bash
archive/dataset_processed_split/
```

The README file provides instructions for:

* source code structure,
* build process,
* installation,
* test execution,
* sample data description,
* data used in the experiment,
* open-source libraries used,
* and experiment result files.

---

## 2. Computational Resource Notice

This repository provides the **full experimental pipeline**, not only a lightweight inference demo.

The code performs:

```text
MTCNN-based face detection
→ ResNet18 feature extraction
→ M-FIG construction
→ feature block clustering
→ Block-QAOA-based feature selection
→ classifier training
→ final evaluation
```

Because the pipeline includes both deep learning-based feature extraction and QAOA-based optimization, execution can be computationally expensive.

CPU-only execution is supported, but it may take a long time depending on the hardware specification and dataset size. For smoother reproduction, a GPU-enabled environment and sufficient disk space are recommended.

Recommended environment:

| Component | Recommendation                                 |
| --------- | ---------------------------------------------- |
| Python    | 3.10                                           |
| Memory    | 16GB RAM or higher recommended                 |
| GPU       | CUDA-supported GPU recommended                 |
| Storage   | Sufficient free disk space required            |
| Runtime   | CPU-only execution is possible but may be slow |

If the execution appears to take a long time, this is likely due to the computational cost of the full pipeline rather than a program error.

---

## 3. Project Description

Deepfake technology can generate or manipulate facial images with high visual realism. As synthetic images become increasingly difficult to distinguish from real images, deepfake detection has become an important task in computer vision and media forensics.

This project proposes a hybrid detection approach that combines **classical deep learning** and **quantum optimization**.

The overall pipeline is as follows:

```text
Input face image
    ↓
Face detection and preprocessing with MTCNN
    ↓
Feature extraction with pretrained ResNet18
    ↓
M-FIG construction
    ↓
Feature block clustering
    ↓
Block-wise feature selection using QAOA
    ↓
Quantum-inspired attention classifier
    ↓
REAL / FAKE prediction
```

The main idea is to first extract high-dimensional visual features from face images and then select more informative and less redundant features using a block-wise QAOA-based optimization method.

---

## 4. Source Code Description

The main source code is implemented in:

```bash
final.py
```

The source code consists of the following major components.

---

### 4.1 Environment and Hyperparameter Settings

At the beginning of the code, dataset paths and major hyperparameters are defined.

```python
BASE_DIR = os.path.join("archive", "dataset_processed_split")
CSV_PATH = os.path.join(BASE_DIR, "dataset_manifest.csv")

NUM_BLOCKS_K = 32
TARGET_FEATURES_TOTAL = 256
LAPLACIAN_THRESHOLD = 100.0
```

| Variable                | Description                                 |
| ----------------------- | ------------------------------------------- |
| `BASE_DIR`              | Base directory of the dataset               |
| `CSV_PATH`              | Path to the dataset manifest file           |
| `NUM_BLOCKS_K`          | Number of feature blocks used in clustering |
| `TARGET_FEATURES_TOTAL` | Number of final selected features           |
| `LAPLACIAN_THRESHOLD`   | Threshold used to filter blurry images      |

The code automatically uses GPU if CUDA is available.

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

If CUDA is not available, the code runs on CPU.

---

### 4.2 Dataset Loading and Preprocessing

The `DeepfakeDataset` class loads image metadata from:

```bash
archive/dataset_processed_split/dataset_manifest.csv
```

For each image, the dataset class performs the following steps:

1. Reads the image file using OpenCV.
2. Converts the image from BGR to RGB.
3. Filters blurry training images using Laplacian variance.
4. Detects and crops the main face using MTCNN.
5. Resizes the original image if face detection fails.
6. Returns the processed face tensor and binary label.

The label conversion rule is:

```text
REAL → 0
FAKE → 1
```

Invalid or unreadable images are safely handled by returning a dummy tensor with label `-1`. These invalid samples are skipped during feature extraction.

---

### 4.3 Feature Extraction with ResNet18

The function `get_backbone()` loads a pretrained ResNet18 model from TorchVision.

The final classification layer is removed, and the model is used only as a feature extractor.

```text
Face image → ResNet18 backbone → 512-dimensional feature vector
```

Each valid input image is converted into a 512-dimensional feature vector.

---

### 4.4 Feature Extraction Function

The function `extract_features()` receives a dataloader and a backbone model.

It returns:

| Output | Description              |
| ------ | ------------------------ |
| `X`    | Extracted feature matrix |
| `y`    | Label vector             |

The function ignores invalid samples whose label is `-1`.

---

### 4.5 M-FIG Construction

The function `build_mfig_and_cluster()` constructs a Multi-Feature Interaction Graph, or M-FIG.

This step calculates:

| Component                | Purpose                                                 |
| ------------------------ | ------------------------------------------------------- |
| Mutual information score | Measures how useful each feature is for classification  |
| Euclidean distance       | Measures similarity between feature activation patterns |
| Pearson correlation      | Measures redundancy between features                    |

The 512-dimensional feature space is then divided into feature blocks using agglomerative clustering.

---

### 4.6 Block-QAOA Feature Selection

The function `select_features_qaoa()` performs block-wise feature selection.

For each feature block, the code formulates a QUBO problem.

The QUBO objective considers:

* selecting features with high mutual information,
* reducing redundant features with high correlation,
* controlling the number of selected features.

QAOA is then used to solve the optimization problem.

If QAOA fails for a certain block, the code automatically switches to a greedy mutual-information-based fallback method.

---

### 4.7 Quantum-inspired Attention Classifier

The final classifier is implemented as:

```python
QuantumHybridClassifier
```

The classifier consists of:

1. Born's rule-inspired trainable probability weights,
2. multi-head attention,
3. fully connected layers,
4. sigmoid output for binary classification.

The output probability is interpreted as:

```text
Probability close to 0 → REAL
Probability close to 1 → FAKE
```

---

### 4.8 Main Execution Pipeline

The `main()` function executes the full experiment.

```text
1. Load train and test datasets
2. Extract ResNet18 features
3. Build M-FIG
4. Cluster features into blocks
5. Select features using Block-QAOA
6. Train the quantum-inspired classifier
7. Evaluate the model on the test set
8. Print final metrics
```

The final evaluation metrics include:

* Accuracy
* Precision
* Recall
* F1-Score
* AUC-ROC

---

## 5. Repository Structure

The repository is expected to have the following structure:

```bash
final/
├── archive/
│   └── dataset_processed_split/
│       ├── train/
│       │   ├── Real/
│       │   ├── DeepFakeDetection/
│       │   ├── Deepfakes/
│       │   ├── Face2Face/
│       │   ├── FaceShifter/
│       │   ├── FaceSwap/
│       │   └── NeuralTextures/
│       │
│       ├── val/
│       │   ├── Real/
│       │   ├── DeepFakeDetection/
│       │   ├── Deepfakes/
│       │   ├── Face2Face/
│       │   ├── FaceShifter/
│       │   ├── FaceSwap/
│       │   └── NeuralTextures/
│       │
│       ├── test/
│       │   ├── Real/
│       │   ├── DeepFakeDetection/
│       │   ├── Deepfakes/
│       │   ├── Face2Face/
│       │   ├── FaceShifter/
│       │   ├── FaceSwap/
│       │   └── NeuralTextures/
│       │
│       └── dataset_manifest.csv
│
├── results/
│   ├── experiment_log.txt
│   └── final_metrics.txt
│
├── final.py
├── requirements.txt
└── Readme.md
```

> Note: The current version of `final.py` mainly uses the `train` and `test` splits. The `val` split is included in the dataset directory but is not directly used in the current code.

---

## 6. How to Build

This project is implemented in Python. Therefore, no separate compilation process is required.

However, users should prepare a Python environment and install all required dependencies before running the code.

Recommended environment:

| Item                | Recommended Setting                      |
| ------------------- | ---------------------------------------- |
| Python              | 3.10                                     |
| Package manager     | pip                                      |
| Virtual environment | conda                                    |
| OS                  | Windows, macOS, or Linux                 |
| GPU                 | Optional, CUDA-supported GPU recommended |

To check whether the Python source code is syntactically valid, run:

```bash
python -m py_compile final.py
```

If no error message appears, the source code has no syntax error.

---

## 7. How to Install

This project can be executed on Windows, macOS, or Linux.

For stable execution, we recommend using:

```text
Python 3.10
conda virtual environment
```

The same `requirements.txt` file is used for all operating systems.

---

### 7.1 Clone the Repository

Clone the repository.

```bash
git clone --depth 1 https://github.com/solmingming/Qubit_research.git
```

Move into the `final` directory.

#### Windows

```bat
cd Qubit_research\final
```

#### macOS / Linux

```bash
cd Qubit_research/final
```

It is important to execute the code inside the `final` directory because the dataset path is defined relative to this location.

---

### 7.2 Create a Virtual Environment

#### Windows

Open **Anaconda Prompt** and run:

```bat
conda create -n qubit-final python=3.10 -y
conda activate qubit-final
```

If the environment is activated successfully, the prompt will look similar to:

```bat
(qubit-final) C:\Users\...\Qubit_research\final>
```

---

#### macOS / Linux

Open Terminal and run:

```bash
conda create -n qubit-final python=3.10 -y
conda activate qubit-final
```

If the environment is activated successfully, the prompt will look similar to:

```bash
(qubit-final) user@computer final %
```

---

### 7.3 Install Required Packages

After activating the environment, upgrade pip first.

```bash
python -m pip install --upgrade pip setuptools wheel
```

Then install the required packages.

```bash
pip install --no-cache-dir -r requirements.txt
```

The `requirements.txt` file should contain:

```txt
numpy==1.26.4
pandas==2.2.2
scipy==1.13.1
pillow==10.4.0

opencv-python-headless==4.10.0.84

scikit-learn==1.5.2
tqdm==4.66.5

torch==2.2.2
torchvision==0.17.2
facenet-pytorch==2.5.3

qiskit>=1.0,<2.0
qiskit-algorithms>=0.3,<0.4
qiskit-optimization>=0.7,<0.8
docplex>=2.25.236
```

This project uses `opencv-python-headless` instead of `opencv-python` to reduce OpenCV GUI-related dependency issues in virtual environments.

---

### 7.4 Installation Check

After installation, check whether the main packages are installed correctly.

```bash
python -c "import cv2; print('cv2 ok:', cv2.__version__)"
python -c "import torch; print('torch ok:', torch.__version__)"
python -c "import torchvision; print('torchvision ok:', torchvision.__version__)"
python -c "import qiskit; print('qiskit ok')"
```

If all commands run without errors, the installation is complete.

---

## 8. How to Test

Before running the full experiment, check whether the source file has no syntax error.

```bash
python -m py_compile final.py
```

If no error message appears, run the full experiment.

```bash
python final.py
```

The program will automatically execute the full pipeline.

```text
Dataset loading
→ Face detection and preprocessing
→ ResNet18 feature extraction
→ M-FIG construction
→ Block-QAOA feature selection
→ Classifier training
→ Final evaluation
```

If the program runs successfully, the terminal will print the final evaluation metrics.

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

A successful run means that:

* the dataset path is correctly set,
* all required packages are installed,
* feature extraction is completed,
* M-FIG construction is completed,
* Block-QAOA feature selection is executed,
* classifier training is completed,
* and final model evaluation is printed.

---

## 9. Experimental Results

Experiment result files are provided in the following directory:

```bash
results/
```

The result directory is intended to include execution logs and final evaluation metrics generated by running `final.py`.

```bash
results/
├── experiment_log.txt
└── final_metrics.txt
```

| File                 | Description                                                                            |
| -------------------- | -------------------------------------------------------------------------------------- |
| `experiment_log.txt` | Full terminal log generated while executing `python final.py`                          |
| `final_metrics.txt`  | Final evaluation metrics, including Accuracy, Precision, Recall, F1-Score, and AUC-ROC |

The final metrics are printed at the end of execution and summarized in `results/final_metrics.txt`.

If the experiment is rerun, the terminal output can be saved as follows.

#### Windows

```bat
mkdir results
python final.py > results\experiment_log.txt 2>&1
```

#### macOS / Linux

```bash
mkdir -p results
python final.py > results/experiment_log.txt 2>&1
```

After execution, the final metric values can be copied from the log file and summarized in:

```bash
results/final_metrics.txt
```

---

## 10. Description of Sample Data

The sample data is stored in:

```bash
archive/dataset_processed_split/
```

The dataset contains three splits:

| Split   | Purpose in Current Code                                                                          |
| ------- | ------------------------------------------------------------------------------------------------ |
| `train` | Used for feature extraction, M-FIG construction, QAOA feature selection, and classifier training |
| `val`   | Included in the dataset, but not directly used in the current version of `final.py`              |
| `test`  | Used for final evaluation                                                                        |

The dataset metadata is stored in:

```bash
archive/dataset_processed_split/dataset_manifest.csv
```

The image path is constructed as:

```bash
archive/dataset_processed_split/{split}/{fake_type}/{filename}
```

Examples:

```bash
archive/dataset_processed_split/train/Real/example.jpg
archive/dataset_processed_split/train/Deepfakes/example.jpg
archive/dataset_processed_split/test/Real/example.jpg
archive/dataset_processed_split/test/FaceSwap/example.jpg
```

---

## 11. Database or Data Used

This project uses a processed deepfake image dataset.

The dataset is organized by image type and split.

Expected fake type folders are:

| Folder Name         | Description                                        |
| ------------------- | -------------------------------------------------- |
| `Real`              | Real face images                                   |
| `DeepFakeDetection` | Fake images from the DeepFakeDetection category    |
| `Deepfakes`         | Fake images generated by the Deepfakes method      |
| `Face2Face`         | Fake images generated by the Face2Face method      |
| `FaceShifter`       | Fake images generated by the FaceShifter method    |
| `FaceSwap`          | Fake images generated by the FaceSwap method       |
| `NeuralTextures`    | Fake images generated by the NeuralTextures method |

The metadata file `dataset_manifest.csv` should contain the following columns:

| Column      | Description                                      |
| ----------- | ------------------------------------------------ |
| `split`     | Dataset split, such as `train`, `val`, or `test` |
| `label`     | Ground-truth label, either `REAL` or `FAKE`      |
| `fake_type` | Folder name of the image category                |
| `filename`  | Image file name                                  |

The current source code uses the `split`, `label`, `fake_type`, and `filename` columns to load images and labels.

---

## 12. Description of Used Open Source

This project uses the following open-source libraries.

| Library             | Purpose                                                       |
| ------------------- | ------------------------------------------------------------- |
| NumPy               | Numerical computation and array processing                    |
| Pandas              | CSV metadata loading and dataframe processing                 |
| SciPy               | Scientific computation dependency used by numerical libraries |
| Pillow              | Image processing dependency used by vision libraries          |
| OpenCV Headless     | Image loading, color conversion, and blur detection           |
| PyTorch             | Neural network implementation and training                    |
| TorchVision         | Pretrained ResNet18 feature extractor                         |
| facenet-pytorch     | MTCNN-based face detection                                    |
| scikit-learn        | Mutual information, clustering, and evaluation metrics        |
| Qiskit              | Quantum computing framework                                   |
| Qiskit Algorithms   | QAOA implementation                                           |
| Qiskit Optimization | QUBO modeling and optimization                                |
| docplex             | Optimization modeling dependency used by Qiskit Optimization  |
| tqdm                | Progress bar visualization                                    |

Major open-source components used in the implementation are described below.

---

### 12.1 TorchVision ResNet18

The project uses a pretrained ResNet18 model from TorchVision as the image feature extraction backbone. The final classification layer is removed so that the model outputs feature vectors instead of class labels.

---

### 12.2 facenet-pytorch MTCNN

MTCNN is used to detect and crop face regions from the input images before feature extraction.

---

### 12.3 Qiskit QAOA

Qiskit-related packages are used to implement the Block-QAOA feature selection process. The feature selection problem is formulated as a QUBO problem and solved using QAOA.

---

### 12.4 OpenCV Headless

OpenCV Headless is used for image loading, RGB conversion, and Laplacian variance-based blur detection. The headless version is used because this project does not require GUI functions such as `cv2.imshow()`.

---

## 13. Reproducibility Notes

To reproduce the experiment, run the code from the `final` directory.

```bash
cd Qubit_research/final
python final.py
```

The first execution may require internet access because TorchVision may download pretrained ResNet18 weights if they are not already cached.

Execution time may vary depending on the environment.

| Environment       | Expected Behavior                                  |
| ----------------- | -------------------------------------------------- |
| CUDA GPU          | Faster feature extraction and training             |
| CPU only          | Supported, but slower                              |
| Apple Silicon Mac | Supported through CPU execution, but may be slower |
| Windows           | Recommended through Anaconda Prompt                |
| Linux server      | Recommended through conda environment              |

Small numerical differences may occur depending on:

* Python version,
* PyTorch version,
* CPU/GPU environment,
* operating system,
* random initialization in model training,
* QAOA optimization behavior.

### Computational Cost

This project may require a relatively high computational cost because it runs the full experimental process rather than only loading a trained model for inference.

The runtime may increase due to:

* MTCNN-based face detection,
* ResNet18 feature extraction,
* feature interaction matrix computation,
* Block-QAOA optimization,
* and classifier training.

CPU-only execution is possible but may be slow. A GPU-enabled environment is recommended for faster reproduction.

---

## 14. Troubleshooting

### 14.1 Dataset Path Error

If the dataset file cannot be found, check whether the following file exists:

```bash
archive/dataset_processed_split/dataset_manifest.csv
```

Also make sure the command is executed from the `final` directory.

```bash
cd Qubit_research/final
python final.py
```

---

### 14.2 Missing Train or Test Data

The current code requires both `train` and `test` splits.

Check whether these folders exist:

```bash
archive/dataset_processed_split/train/
archive/dataset_processed_split/test/
```

Also check whether `dataset_manifest.csv` contains rows where the `split` column is `train` and `test`.

---

### 14.3 CSV Column Error

If a key error occurs while reading the CSV file, check whether `dataset_manifest.csv` contains the following columns:

```text
split
label
fake_type
filename
```

---

### 14.4 ModuleNotFoundError

If a Python package is missing, install dependencies again.

```bash
pip install --no-cache-dir -r requirements.txt
```

---

### 14.5 OpenCV Import Error

If `cv2` cannot be imported, reinstall OpenCV Headless.

```bash
pip uninstall -y opencv-python opencv-python-headless
pip install opencv-python-headless==4.10.0.84
```

Then test:

```bash
python -c "import cv2; print(cv2.__version__)"
```

---

### 14.6 PyTorch Installation Error

If PyTorch installation fails, install PyTorch separately according to the local CPU or GPU environment.

For CPU-only execution, the default pip installation is usually sufficient.

```bash
pip install torch==2.2.2 torchvision==0.17.2
```

Then reinstall the remaining packages.

```bash
pip install --no-cache-dir -r requirements.txt
```

---

### 14.7 Qiskit-related Error

If Qiskit-related modules are missing, reinstall the Qiskit packages.

```bash
pip install "qiskit>=1.0,<2.0" "qiskit-algorithms>=0.3,<0.4" "qiskit-optimization>=0.7,<0.8" "docplex>=2.25.236"
```

---

### 14.8 No Space Left on Device

If the following error appears:

```text
OSError: [Errno 28] No space left on device
```

there is not enough disk space to install the dependencies.

Recommended actions:

* remove unused conda environments,
* clear pip cache,
* clear conda cache,
* free several GB of disk space,
* reinstall with `--no-cache-dir`.

Example:

```bash
conda clean --all -y
pip cache purge
pip install --no-cache-dir -r requirements.txt
```

---

### 14.9 Slow Execution

This code may take time because it performs:

* MTCNN face detection,
* ResNet18 feature extraction,
* feature interaction calculation,
* QAOA-based optimization,
* neural network training.

CPU-only execution is supported but may be slow.

---

## 15. Team

**Team Qubit**

| Name | Role                           |
| ---- | ------------------------------ |
| 이솔민  | Graduation project team leader |
| 김은솜  | Graduation project team member |
| 김정민  | Graduation project team member |

Graduation Project
Department of Computer Science and Engineering
Ewha Womans University

---

## 16. License

This repository is intended for academic and educational purposes.

If you use this repository, please refer to the source code and documentation included in this project.
