# GPT-2 Fine-tuning with PPO on IMDb Dataset

## Overview

The project utilizes the IMDb dataset to provide relevant training examples for the GPT-2 model, with adjustments made via PPO to optimize performance based on human feedback mechanisms.

## Files Description

- **dataset.py**: Handles loading and preprocessing the IMDb dataset for use in model training.
- **ppo_training.py**: Implements the PPO training algorithm, adapted for fine-tuning the GPT-2 model.
- **train_reward.py**: Computes and adjusts the reward signals during training based on model outputs and target feedback.
- **run_ppo.sh**: A shell script to facilitate the model training process.

## How to Run

To begin training the GPT-2 model with the IMDb dataset using PPO, execute the provided shell script:

```bash
./run_ppo.sh



## Training Results

Training progress and outcomes are visualized through the following plots:

- **kl_divergence.png**: Displays the KL divergence, offering insights into the divergence between the policy and target distributions over training. From the picture, the KL divergence trend is steadily increasing, suggesting that the model's policy is continuously learning while maintaining a controlled divergence from the baseline policy.
- **ppo_loss.png**: Shows the PPO loss, tracking how well the model minimizes its error across training epochs. From the picture, the PPO loss decreases over time, which signifies that the model is effectively learning and optimizing its predictions throughout the training.
- **ppo_reward.png**: Plots the reward trajectory, demonstrating the effectiveness of the training process over time. From the picture, the mean reward graph shows a positive and stable increase, indicating that the model's performance is consistently improving.
