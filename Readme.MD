<!-- HEADER -->
<div align="center">

  <h1>딥러닝을 활용한 양자 노이즈 완화</h1>
  <p><em>Quantum Noise Mitigation with Deep Learning</em></p>

  <!-- 핵심 배지 -->
  <p>
    <img src="https://img.shields.io/badge/Qiskit-6B4C9A?logo=qiskit&logoColor=white" alt="Qiskit"/>
    <img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch"/>
    <img src="https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white" alt="Python"/>
    <img src="https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white" alt="Jupyter"/>
  </p>

  <!-- 네비게이션 -->
  <p>
    <a href="#-프로젝트-소개">소개</a> ·
    <a href="#-project-overview-english">English</a> ·
    <a href="#-실험-설계--아키텍처">아키텍처</a> ·
    <a href="#-members">Members</a> ·

  </p>

  <!-- 티저 이미지 (나중에 결과 나오면 추가) -->
  <img src="./docs/teaser.png" alt="Project Teaser" width="720"/>
</div>

---

## 📌 프로젝트 소개
팀명 <b>Qubit</b>은 양자컴퓨팅의 최소 단위이자  
여러 가능성을 동시에 품는 <b>중첩(Superposition)</b> 개념에서 영감을 얻었습니다.

우리 팀은 <b>다양한 역량과 잠재력</b>을 결합해  
<b>실험적 탐구</b>와 <b>성장</b>을 최우선으로 하는 프로젝트를 만들어 갑니다.

**주요 연구 내용**
- ⚡ Qiskit으로 노이즈 포함 양자 회로 데이터 생성  
- 🧩 Autoencoder / VAE 모델로 상태 복원  
- 📊 Fidelity, Trace distance 지표로 성능 평가  

> **목표:** 학부 수준에서 **딥러닝 기반 노이즈 완화**의 실험적 가능성 검증

---

## 🌍 Project Overview (English)
Our team name <b>“Qubit”</b> is inspired by the concept of <b>superposition</b>,  
where a qubit can exist in multiple states at once.  

We aim to integrate <b>versatile skills</b> of our members  
to pursue meaningful outcomes through <b>evidence-driven research</b>.

**Focus**
- ⚡ Generate noisy quantum circuit data via Qiskit  
- 🧩 Restore original states using Autoencoder / VAE  
- 📊 Evaluate using fidelity & trace distance  

---

## ⏳ 프로젝트 기간
<b>2025.09 ~ 2026.07</b>

---

## 🧪 실험 설계 & 아키텍처 (Draft)
<details>
<summary><b>파이프라인 개요</b></summary>

Ideal Circuits ─> Noise Injection ─> Noisy Data
│
┌────────────┴────────────┐
│ │
Train Split Test Split
│ │
Model Training Model Evaluation

</details>

> 이후 모델 구조 다이어그램, 결과 그래프 등을 `docs/`에 추가 후 여기 표시 예정

---

## 👩‍💻 Members
<table>
  <thead>
    <tr>
      <th>이름</th>
      <th>역할</th>
      <th>소속/학번</th>
      <th>Tech Stack</th>
      <th>CV</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>이솔민</b></td>
      <td>Leader</td>
      <td>이화여자대학교 컴퓨터공학과 22 / 경영 복수전공</td>
      <td><code>HTML</code> <code>CSS</code> <code>React</code> <code>Tailwind</code> <code>C</code> <code>JavaScript</code></td>
      <td>
        <a href="./docs/solmin Lee_cv.pdf" target="_blank">
          <img alt="Download CV" src="https://img.shields.io/badge/CV-Download-blue">
        </a>
      </td>
    </tr>
    <tr>
      <td><b>김은솜</b></td>
      <td>Member</td>
      <td>이화여자대학교 컴퓨터공학과 22</td>
      <td><code>Java</code> <code>Python</code> <code>C++</code></td>
      <td>
        <a href="./docs/eunsom Kim_cv.pdf" target="_blank">
          <img alt="Download CV" src="https://img.shields.io/badge/CV-Download-blue">
        </a>
      </td>
    </tr>
    <tr>
      <td><b>김정민</b></td>
      <td>Member</td>
      <td>이화여자대학교 컴퓨터공학과 22</td>
      <td><code>HTML</code> <code>CSS</code> <code>C++</code></td>
      <td>
        <a href="./docs/jeongmin Kim_cv.pdf" target="_blank">
          <img alt="Download CV" src="https://img.shields.io/badge/CV-Download-blue">
        </a>
      </td>
    </tr>
  </tbody>
</table>

---

## 🗺️ Roadmap
- [ ] 데이터셋/노이즈 시뮬레이션 초기 버전 구축  
- [ ] Autoencoder/Variational Autoencoder 학습  
- [ ] Fidelity·Trace distance 비교 실험  
- [ ] 결과 시각화 & 보고서 작성  
- [ ] 최종 포스터/발표 준비  

---

<p align="center">
  <sub>© 2025 Team Qubit — All Rights Reserved</sub><br/>
  <a href="./docs/">📁 Browse docs/</a>
</p>

