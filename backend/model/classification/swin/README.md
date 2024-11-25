---
license: apache-2.0
base_model: microsoft/swin-tiny-patch4-window7-224
tags:
- generated_from_trainer
metrics:
- precision
- recall
- f1
- accuracy
model-index:
- name: swin-tiny-patch4-window7-224-musinsa
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# swin-tiny-patch4-window7-224-musinsa

This model is a fine-tuned version of [microsoft/swin-tiny-patch4-window7-224](https://huggingface.co/microsoft/swin-tiny-patch4-window7-224) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 0.1442
- Precision: 0.9553
- Recall: 0.9552
- F1: 0.9551
- Accuracy: 0.9552

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 128
- eval_batch_size: 128
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 10

### Training results

| Training Loss | Epoch | Step | Validation Loss | Precision | Recall | F1     | Accuracy |
|:-------------:|:-----:|:----:|:---------------:|:---------:|:------:|:------:|:--------:|
| No log        | 1.0   | 202  | 0.1506          | 0.9427    | 0.9423 | 0.9424 | 0.9423   |
| No log        | 2.0   | 404  | 0.1260          | 0.9549    | 0.9548 | 0.9548 | 0.9548   |
| 0.158         | 3.0   | 606  | 0.1318          | 0.9561    | 0.9562 | 0.9561 | 0.9562   |
| 0.158         | 4.0   | 808  | 0.1442          | 0.9553    | 0.9552 | 0.9551 | 0.9552   |


### Framework versions

- Transformers 4.41.2
- Pytorch 2.3.1+cu121
- Tokenizers 0.19.1
