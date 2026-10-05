# CryptOQA @ FIRE 2026

This repository contains the implementation of **Team TokenX** for the **CryptOQA @ FIRE 2026** shared task, *Understanding CryptoCurrency-related Opinions and Questions from Social Media Posts*.

The system addresses two tasks:

* **Task 1:** Opinion Classification from CryptoCurrency-related Social Media Posts
* **Task 2:** Question Answering from CryptoCurrency-related Social Media Posts

## Pipeline Overview

### Task 1: Hierarchical Opinion Classification

The Task 1 pipeline follows the hierarchical structure of the dataset and trains separate classifiers for the three classification levels.

```text
Social Media Posts
        |
        v
   Preprocessing
        |
        v
 Train / Validation Split
        |
        v
 Transformer Models
        |
        +-------------------------+
        |            |            |
        v            v            v
     Level 1      Level 2      Level 3
        |            |            |
        +------------+------------+
                     |
                     v
             Stacking Ensemble
                     |
                     v
             Final Predictions
```

Three pretrained transformer encoders are evaluated:

* DeBERTa
* RoBERTa
* DistilRoBERTa

The training pipeline includes:

* Unicode normalization and text cleaning
* Whitespace normalization
* Duplicate removal
* Hierarchical label preparation
* Class-weighted cross-entropy
* Label smoothing
* Gradient accumulation
* Gradient clipping
* Learning-rate warm-up
* Mixed-precision computation
* Gradient checkpointing

The individual model predictions are combined using a **stacking ensemble**.

### Task 2: YouTube-based Question Answering

Task 2 focuses on **candidate-based Multiple-Choice Question Answering (MCQA)** using YouTube video transcripts.

```text
Question
    |
    +----------------------+
    |                      |
    v                      v
Video Transcript      Candidate Answers
    |                      |
    +----------+-----------+
               |
               v
      Question + Transcript
        + Candidate
               |
               v
       Transformer Encoder
               |
               v
        Candidate Scores
               |
               v
     Highest-scoring Candidate
               |
               v
        Predicted Answer
```

For each question, the associated video transcript is combined with each candidate answer. The transformer encoder assigns a score to each candidate, and the candidate with the highest score is selected.

The MCQA questions are categorized into:

* **Single-hop:** The answer can be determined using information from a single relevant piece of contextual evidence.
* **Multi-hop:** The answer requires combining information from multiple pieces of contextual evidence.

The evaluated transformer encoders are:

* RoBERTa
* DeBERTa

## Models

| Task   | Models                          |
| ------ | ------------------------------- |
| Task 1 | DeBERTa, RoBERTa, DistilRoBERTa |
| Task 2 | RoBERTa, DeBERTa                |

## Task 1 Hierarchical Labels

Task 1 uses three classification levels.

### Level 1

| Label | Description |
| ----- | ----------- |
| 0     | NOISE       |
| 1     | OBJECTIVE   |
| 2     | SUBJECTIVE  |

### Level 2

Applied within the subjective category:

| Label | Description |
| ----- | ----------- |
| 0     | NEUTRAL     |
| 1     | NEGATIVE    |
| 2     | POSITIVE    |

### Level 3

Applied within the neutral category:

| Label | Description       |
| ----- | ----------------- |
| 0     | NEUTRAL SENTIMENT |
| 1     | QUESTIONS         |
| 2     | ADVERTISEMENTS    |
| 3     | MISCELLANEOUS     |

## Experimental Configuration

### Task 1

* Learning rate: `2e-5`
* Batch size: `16`
* Epochs: `3`
* Maximum sequence length: `128`
* Weight decay: `0.01`
* Warm-up ratio: `0.06`
* Label smoothing: `0.05`
* Gradient clipping: `1.0`
* Gradient accumulation: `2`

### Task 2 MCQA

* Learning rate: `2e-5`
* Batch size: `4`
* Epochs: `3`
* Maximum sequence length: `256`
* Weight decay: `0.01`
* Warm-up ratio: `0.06`
* Label smoothing: `0.05`
* Gradient clipping: `1.0`
* Gradient accumulation: `4`

## Results

### Task 1

The stacking ensemble improves validation performance across all three hierarchical classification levels.

The official submitted system achieved:

* **Accuracy: 88.34%**
* **Rank: 1**

### Task 2

The selected RoBERTa model achieved the following official MCQA results:

| Evaluation      | Accuracy |
| --------------- | -------: |
| Random baseline |    25.0% |
| Overall         |    22.8% |
| Single-hop      |    24.0% |
| Multi-hop       |    22.5% |

The random baseline corresponds to four candidate answers.

## Dataset

### Task 1

The system uses cryptocurrency-related social media posts from:

* Reddit
* Twitter
* YouTube

### Task 2

The system uses YouTube-based question answering data together with associated video transcripts and candidate answers.

The video collection contains **974 videos**, including **23 videos with empty transcripts**. The transcripts can be substantially longer than the model input limit, with a mean length of approximately **3,619.9 words** and a maximum length of **35,588 words**.

## Repository Structure

The main implementation is provided in the repository through the accompanying notebook and supporting files.

```text
CryptOQA-2026/
│
├── CRYPTO.ipynb
├── README.md
└── ...
```

## Reproducibility

The experiments use a fixed random seed of `42`. The implementation is designed for GPU-based execution using CUDA and supports mixed-precision training and gradient checkpointing.

The main experimental workflow is contained in:

```text
CRYPTO.ipynb
```

The notebook contains the preprocessing, model training, validation, ensemble construction, prediction generation, and submission-related processing.

## Team

**Team TokenX**

CryptOQA @ FIRE 2026

## Citation

If you use this implementation or refer to the system in your work, please cite the corresponding CryptOQA @ FIRE 2026 system description paper.

## Repository

The complete implementation is publicly available at:

[https://github.com/rachanabn20/CryptOQA-2026](https://github.com/rachanabn20/CryptOQA-2026)

