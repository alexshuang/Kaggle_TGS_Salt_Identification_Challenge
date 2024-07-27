## [[TGS Salt Identification Challenge](https://www.kaggle.com/c/tgs-salt-identification-challenge)] Bronze Medal Soultion

First of all, Thanks [Fastai library](https://github.com/fastai/fastai/tree/0.7.0/fastai) which save me a lot of time to build the model. Bronze medal place isn't a very good score, However, this [notebook](https://github.com/alexshuang/TGS_Salt/blob/master/TGS_Salt_resnext50_unet_5Fold_scSE.ipynb) can be directly run on Google colab. You can use it as backbone to experiment with the ideas published by other top contestants. The most important thing is that you don't need to have your own GPU for this competition, you'd better have a faster GPU, though:).

For more information about source code, please read my chinese blog: https://www.jianshu.com/p/ab0a10c2e710

---

### Network architecture

* U-net with a pre-trained resnet34/resnext50, resnext50 is better.

### Training regime

* Adam optimizer with weight decay. SGD is better than Adam, but it also much lower than Adam.
* 5-folds cross validation.
* use [albumentations](https://github.com/albu/albumentations) library to do data augmentation, random corp 50%~100% area region, scale to 128x128. Scale to 192x192 or 256x256 is recommended, but you need to change to a faster GPU.
* stage1: train by SGD + BCE, stage2: train by Adam + Lovasz Hinge Loss, until validation loss is no longer reduced.

### Techniques that helped

* Adding Concurrent Spatial and Channel Squeeze & Excitation blocks to the U-net decoder **(+0.032 Private LB)**: [https://arxiv.org/pdf/1803.02579.pdf.](https://arxiv.org/pdf/1803.02579.pdf.) I used the [implementation](https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/66178) generously provided by [Bilal](https://www.kaggle.com/bkkaggle).
* Depth-statified n-fold training (+0.01 LB). For me, 5-fold scored is good.
* TTA: flip and non-flip (+0.02 LB)

### Techniques that not helped
*   Hypercolumns: [https://arxiv.org/pdf/1411.5752.pdf.](https://arxiv.org/pdf/1411.5752.pdf.)
*   Many folks([Heng](https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/64645#380301), et al) reported anywhere from a public LB +0.01 to +0.005 improvement after concatenating hypercolumns to their decoder's output. My attempt led to a lower score, and because I found it to be computationally expensive on my Paperspace P5000 machine, I didn't bother seeing if I could troubleshoot.

