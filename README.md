# ETF-Enhanced Pseudo Feature Approach for Federated Class-Incremental Learning Without Rehearsal

## Introduction
This repository contains the implementation of **FLEX (Federated Class-Incremental Learning with ETF Vectors and Pseudo Feature EXtension)**, a novel framework designed to address **catastrophic forgetting** in **federated class-incremental learning (FCIL)** without using rehearsal memory.

FLEX integrates:
- **Equiangular Tight Frame (ETF) vectors** for structured **Data-Free Knowledge Transfer (DFKT)**.
- **Pseudo features** that allow the **fully connected (FC) layer** to continue learning after freezing the feature extractor.
- A design that significantly reduces **communication costs** while maintaining **state-of-the-art performance**.

<p align="center">
  <img src="./img/figure1.png" alt="FLEX Framework" width="600"/>
</p>

**Figure 1**: Overview of the FLEX framework. The server uses ETF vectors for data-free knowledge transfer and generates pseudo features. The client integrates pseudo features, bounding loss, and knowledge distillation for efficient federated class-incremental learning.

---

## Key Features

- **Efficient Data-Free Knowledge Transfer (DFKT)** with ETF vectors, eliminating the need for vision-language models (VLMs).
- **Integration of Pseudo Features** for training only the FC layer, reducing computational complexity.
- **Lower Communication Cost**: By freezing the feature extractor and exchanging only the FC layer with the server, communication overhead is reduced.
- **Superior Performance**: Outperforms existing FCIL methods in **non-IID federated learning environments**.

---

## Experimental Results

We conducted experiments on **CIFAR-100** and **TinyImageNet** under various **non-IID conditions**. FLEX achieved **state-of-the-art performance**, outperforming previous methods.

### 1️⃣ Performance Comparison Across Non-IID Settings

The table below presents the performance comparison for **T=5 tasks** on CIFAR-100 under different levels of non-IID distribution:

| Method  | IID  | NIID (β=1) | NIID (β=0.5) | NIID (β=0.3) | NIID (β=0.1) |
|---------|------|-----------|------------|------------|------------|
| **FLEX**  | 62.25 | 64.46  | **62.55**  | **61.81**  | **58.75**  |
| LANDER  | **66.26** | 64.94  | 61.33  | 60.02  | 57.37  |
| TARGET  | 47.77 | 49.36  | 39.56  | 44.59  | 41.67  |
| FedLWF  | 58.07 | 54.72  | 52.41  | 51.78  | 48.77  |
| FedEWC  | 41.00 | 40.51  | 39.36  | 36.93  | 35.34  |

**Table 1**: Performance Comparison under IID and Non-IID Scenarios. FLEX consistently outperforms existing methods in non-IID settings.

### 2️⃣ Task-wise Performance Under Different Non-IID Settings

<p align="center">
  <img src="./img/figure2.png" alt="Task-wise Performance" width="600"/>
</p>

**Figure 2**: Task-wise performance comparison across different non-IID levels (β=1, β=0.5, β=0.3, β=0.1). FLEX demonstrates superior performance across most tasks, particularly in highly non-IID settings.

### 3️⃣ Communication Cost Reduction

One of FLEX’s major advantages is its **significant reduction in communication costs**. By freezing the feature extractor and only transmitting the **fully connected (FC) layer**, FLEX **reduces parameter exchange by approximately 20% per task**.

<p align="center">
  <img src="./img/figure4.png" alt="Communication Cost Reduction" width="600"/>
</p>

**Figure 4**: FLEX achieves **20% communication cost reduction per task** compared to existing methods.

---

## Installation & Usage

### Prerequisites
- Python 3.8+
- PyTorch 1.10+
- CUDA (for GPU acceleration)
- torchvision, numpy, tqdm

### Installation
```bash
git clone https://github.com/minkyoon/2024_Master_Thesis_FLEX_FCILL.git
cd 2024_Master_Thesis_FLEX_FCIL
pip install -r requirements.txt
```

### Traing
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --group=c100t5 --exp_name=lander_b05 --dataset cifar100 --method=flex --tasks=5 --num_users 5 --beta=0.5 
```