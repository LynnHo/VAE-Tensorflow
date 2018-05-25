# VAE

Tensorflow implementation of VAE.

VAE: [Auto-encoding variational bayes](http://arxiv.org/abs/1312.6114)

## Exemplar results

### Celeba
Generation

<img src="./pics/celeba_generate_Epoch_(99)_(3165of3165).jpg">

Reconstruction

<img src="./pics/celeba_reconstruct_Epoch_(99)_(3165of3165).jpg">

### Mnist
Generation

<img src="./pics/mnist_generate_Epoch_(99)_(937of937).jpg">

Reconstruction

<img src="./pics/mnist_reconstruct_Epoch_(99)_(937of937).jpg">

# Prerequisites
- tensorflow r1.2
- python 2.7

# Usage

## Train
```
python train_celeba_vae.py
python train_mnist_vae.py
```
## Tensorboard
```
tensorboard --logdir=./summaries/celeba --port=6006
...
```

## Datasets
1. Mnist will be automatically downloaded
2. Celeba should be prepared by yourself in ./data/img_align_celeba/*.jpg
    - Download the dataset: https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AAB06FXaQRUNtjW9ntaoPGvCa?dl=0
    - the above links might be inaccessible, the alternative is
        - ***img_align_celeba.zip***
            - https://pan.baidu.com/s/1eSNpdRG#list/path=%2FCelebA%2FImg or
            - https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg