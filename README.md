# Transformer Training Optimization – DistilBERT

## Project Overview

This project studies the impact of optimization hyperparameters on Transformer model fine-tuning.

The experiments use a pretrained DistilBERT model fine-tuned on the SST-2 sentiment classification dataset.

The main objective is to analyze how different Learning Rates affect model performance and training time.

## Student 1 – Dataset & Baseline

My contribution focuses on:
- Dataset selection and loading.
- Data cleaning and preprocessing.
- Train, Validation and Test preparation.
- Text tokenization using Hugging Face.
- DistilBERT model loading and fine-tuning.
- Baseline training.
- Learning Rate comparison.

## Dataset

- **Dataset:** SST-2 (GLUE)
- **Task:** Binary sentiment classification
- **Labels:** 0 = Negative, 1 = Positive
- **Training samples:** 6,000
- **Validation samples:** 1,000
- **Test samples:** 870

The official GLUE test split does not provide public labels. Therefore, the official validation split is used as the held-out test set in this experiment.

## Technologies

- Python
- PyTorch
- Hugging Face Datasets
- Hugging Face Transformers
- Pandas
- Scikit-learn
- Google Colab

## Baseline Configuration

| Parameter | Value |
|---|---|
| Model | DistilBERT |
| Learning Rate | 2e-5 |
| Optimizer | AdamW |
| Batch Size | 32 |
| Epochs | 3 |
| Scheduler | Linear |
| Warmup | 0 |
| Max Sequence Length | 64 |

## Learning Rate Experiments

Four Learning Rates were compared while keeping the other configuration parameters fixed.

| Learning Rate | Validation Accuracy | Validation F1 |
|---|---:|---:|
| 1e-5 | 88.50% | 88.42% |
| 2e-5 | 89.70% | 89.63% |
| 5e-5 | 91.50% | 91.43% |
| 1e-4 | **91.60%** | **91.53%** |

The best Learning Rate on the validation set was **1e-4**.

## Final Test Results

| Metric | Result |
|---|---:|
| Accuracy | 88.28% |
| F1-score | 88.27% |
| Precision | 88.27% |
| Recall | 88.28% |

## Training Time

All training runs were completed within the project's 10-minute limit per run.

## Repository Structure

```text
transformer-optimization-distilbert/
├── Dataset_Baseline.ipynb
├── README.md
├── requirements.txt
├── data/
├── results/
├── models/
└── reports/
```

## Notes

This repository contains the Student 1 contribution: Dataset preparation, preprocessing, baseline training and Learning Rate experiments.

Further experiments on warmup, scheduler, batch size and optimizer comparison are part of the other project contributions.
