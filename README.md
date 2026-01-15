# Deepfake Image Classification

This project was developed for a Kaggle competition focused on classifying AI-generated images. I achieved 5th place overall out of all participants.

[Competition Leaderboard](https://www.kaggle.com/competitions/deepfake-classification-unibuc/leaderboard)

## Overview

The task involved multi-class classification of over 12,000 images across five deepfake categories. The goal was to distinguish between real images and various types of AI-generated deepfakes.

## Results

- Final ranking: 5th place
- Validation accuracy: 94.3%

## Approach

I experimented with two different approaches:

**K-Nearest Neighbors (KNN)**: A baseline implementation using feature extraction and distance-based classification.

**Convolutional Neural Network (CNN)**: The final model architecture used residual blocks and several modern techniques:

- CutMix augmentation
- Label smoothing
- Test Time Augmentation (TTA)

The CNN model significantly outperformed the KNN baseline and was used for the final submission.

## Project Structure

```
├── cnn.py              # CNN implementation with residual blocks
├── knn.py              # KNN baseline implementation
├── train.csv           # Training dataset labels
├── validation.csv      # Validation dataset labels
├── test.csv            # Test dataset for predictions
├── submission.csv      # Final competition submission
├── train/              # Training images
├── validation/         # Validation images
└── test/               # Test images
```
