## Segmentation

![Figure 1 ](https://upload-images.jianshu.io/upload_images/13575947-e7d482d05e62cb96.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
Figure1来自CamVid database，专为目标识别（Object Dection）和图像分割（Image Segmentation）提供训练数据的网站。从图中可以看出，segmentation将图像中不同个体用不同颜色来标记，这里不同的颜色就代表不同的分类，例如红色就是分类1，蓝色就是分类2，可以看出，它就是像素级的图像识别（Image Identification）。

除了自动驾驶之外，图像分割还广泛应用于医学诊断、卫星影像定位、图片合成等领域，本文就以当前[kaggle](http://www.kaggle.com)上最热门的segmentation竞赛--[TGS Salt Identification Challenge](https://www.kaggle.com/c/tgs-salt-identification-challenge)为例来讲解如何应用Unet来解决真实世界的图像分割问题。github: [here](https://github.com/alexshuang/TGS_Salt)。

TGS公司通过地震波反射技术绘制出下图所示的3D地质图像，并标记出图像中的盐矿区域，参赛者需要训练用于从岩层中分离盐矿的机器学习模型。
![Figure 2: Images & marks](https://upload-images.jianshu.io/upload_images/13575947-ffe482168fdeff0f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

Figure 2是trainset中5组image和mark图片，每组的左图为原始地质图像，右图为该图像的分类，称为mark，黑色区域代表一般岩层，白色区域就是盐的分布。segmentation要做的就是训练一个image-to-image的模型，通过对原始图像的学习，生成其对应的mask<sub>2</sub>，mask则作为target，通过最小化mask和mask<sub>2</sub>的差距来识别哪些是盐。

### Dataset

生成dataset的第一步是根据run length数据创建对应的mark图片，因为TGS的trainset里面已经提供了mark图片（mark图片和对应image图片同名），所以我们就不需要额外再创建。

但要知道的是，并非所有的segmentation dataset都会提供marks，你需要根据数据run length来为images创建相应的marks，run length是如下图rle_mask所示的数据，数据间以空格分隔，两两为一组，每组的第一个数代表flatten后的image vector的起始下标，后一个数代表它所占据的长度，占据区域会填充该目标对应的分类号，如0、1、2...，通过rle_decode()可以将run length转化为mark。
![image.png](https://upload-images.jianshu.io/upload_images/13575947-376d4fda8d49f757.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

```
def rle_decode(mask_rle, shape=(101, 101)):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.mean()
```

从Figure 2可以看到，地质图像都是低分辨画质，只有101x101大小，不仅不利于神经网络的卷积计算，也不利于图像识别，所以我们接下来一般会将其resize为128x128大小。
```
def resize_img(fn, outp_dir, sz):
  Image.open(fn).resize((sz, sz)).save(outp_dir/fn.name)
```
Data augmentation是创建dataset的核心，和object dection一样，segmentation一般不会做random crop，我在这个项目中采用水平、垂直翻转和微调光暗的方法来做augmentation。
```
aug_tfms = [
    RandomFlip(tfm_y=TfmType.CLASS),
    RandomDihedral(tfm_y=TfmType.CLASS),
#     RandomRotate(4, tfm_y=TfmType.CLASS),
    RandomLighting(0.07, 0.07, tfm_y=TfmType.CLASS)
]
```

### Unet
[paper](https://arxiv.org/abs/1505.04597)
Unet虽然是2015年诞生的模型，但它依旧是当前segmentation项目中应用最广的模型，kaggle上LB排名靠前的选手很多都是使用该模型。
![image.png](https://upload-images.jianshu.io/upload_images/13575947-2ce6b66cc5d3df89.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
Unet的左侧是convolution layers，右侧则是upsamping layers，convolutions layers中每个pooling layer前一刻的activation值会concatenate到对应的upsamping层的activation值中。

因为Unet左侧部分和resnet、vgg、inception等模型一样，都是通过卷积层来提取图像特征，所以Unet可以采用resnet/vgg/inception+upsampling的形式来实现，这样做好处是可以利用pretrained的成熟模型来加速Unet的训练，要知道transfer training的效果是非常显著的，我在这个项目中采用的就是resnet34+upsampling的架构。
```
class SaveFeatures():
    features=None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = output
    def remove(self): self.hook.remove()


class UnetBlock(nn.Module):
  def __init__(self, up_in, down_in, n_out, dp=False, ps=0.25):
    super().__init__()
    up_out = down_out = n_out // 2
    self.tr_conv = nn.ConvTranspose2d(up_in, up_out, 2, 2, bias=False)
    self.conv = nn.Conv2d(down_in, down_out, 1, bias=False)
    self.bn = nn.BatchNorm2d(n_out)
    self.dp = dp
    if dp: self.dropout = nn.Dropout(ps, inplace=True)
  
  def forward(self, up_x, down_x):
    x1 = self.tr_conv(up_x)
    x2 = self.conv(down_x)
    x = torch.cat([x1, x2], dim=1)
    x = self.bn(F.relu(x))
    return self.dropout(x) if self.dp else x


class Unet34(nn.Module):
  def __init__(self, rn, drop_i=False, ps_i=None, drop_up=False, ps=None):
    super().__init__()
    self.rn = rn
    self.sfs = [SaveFeatures(rn[i]) for i in [2, 4, 5, 6]]
    self.drop_i = drop_i
    if drop_i:
      self.dropout = nn.Dropout(ps_i, inplace=True)
    if ps_i is None: ps_i = 0.1
    if ps is not None: assert len(ps) == 4
    if ps is None: ps = [0.1] * 4
    self.up1 = UnetBlock(512, 256, 256, drop_up, ps[0])
    self.up2 = UnetBlock(256, 128, 256, drop_up, ps[1])
    self.up3 = UnetBlock(256, 64, 256, drop_up, ps[2])
    self.up4 = UnetBlock(256, 64, 256, drop_up, ps[3])
    self.up5 = nn.ConvTranspose2d(256, 1, 2, 2)
  
  def forward(self, x):
    x = F.relu(self.rn(x))
    x = self.dropout(x) if self.drop_i else x
    x = self.up1(x, self.sfs[3].features)
    x = self.up2(x, self.sfs[2].features)
    x = self.up3(x, self.sfs[1].features)
    x = self.up4(x, self.sfs[0].features)
    x = self.up5(x)
    return x[:, 0]
  
  def close(self):
    for o in self.sfs: o.remove()
```
通过注册nn.register_forward_hook() ，将指定resnet34指定层（2, 4, 5, 6）的activation值保存起来，在upsampling的过程中将它们concatnate到相应的upsampling layer中。upsampling layer中使用ConvTranspose2d()来做deconvolution，ConvTranspose2d()的工作机制和conv2d()正好相反，用于增加feature map的grid size，对deconvolution的计算不是很熟悉的朋友请自行阅读[convolution arithmetic tutorial](http://deeplearning.net/software/theano/tutorial/conv_arithmetic.html)。

### Loss

前文也提到，segmentation本质上是像素级的图像识别，该项目只有两个分类: 盐和岩，和猫vs狗一样是binary classification问题，用binary cross entropy即可，即nn.BCEWithLogitsLoss()。除了BCE，我还尝试了[focal loss](https://arxiv.org/abs/1708.02002)，准确率提升了0.013。
![Figure 3](https://upload-images.jianshu.io/upload_images/13575947-ff420a35c83e81e8.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

从Figure 3中数学公式可以看出，focal loss就是scale版的cross entropy，$-(1 - p_t)^\gamma$是scale值，这里的scale不是常数而是可学习的weights。focal loss的公式虽然很简单，但在object dection中，focal loss的表现远胜于BCE，其背后的逻辑是：通过scale放大/缩小模型的输出结果，将原本模糊不清的判断确定化。Figure 3，当gamma == 0时，focal loss就相当于corss entropy(CE)，如蓝色曲线所示，即使probability达到0.6，loss值还会>= 0.5，就好像是说：“我判断输出不是分类B的概率是60%，但我依旧不能确定它一定不是分类B”。当gamma == 2时，同样是probability达到0.6，loss值接近于0，就好像是说：“我判断输出不是分类B的概率是60%，我认为它一定不是分类B”，这就是scale的威力。
```
#https://github.com/marvis/pytorch-yolo2/blob/master/FocalLoss.py
#https://github.com/unsky/focal-loss
class FocalLoss2d(nn.Module):
    def __init__(self, gamma=2, size_average=True):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, logit, target, class_weight=None, type='softmax'):
        target = target.view(-1, 1).long()
        if type=='sigmoid':
            if class_weight is None:
                class_weight = [1]*2 #[0.5, 0.5]
            prob   = F.sigmoid(logit)
            prob   = prob.view(-1, 1)
            prob   = torch.cat((1-prob, prob), 1)
            select = torch.FloatTensor(len(prob), 2).zero_().cuda()
            select.scatter_(1, target, 1.)
        elif  type=='softmax':
            B,C,H,W = logit.size()
            if class_weight is None:
                class_weight =[1]*C #[1/C]*C
            logit   = logit.permute(0, 2, 3, 1).contiguous().view(-1, C)
            prob    = F.softmax(logit,1)
            select  = torch.FloatTensor(len(prob), C).zero_().cuda()
            select.scatter_(1, target, 1.)
        class_weight = torch.FloatTensor(class_weight).cuda().view(-1,1)
        class_weight = torch.gather(class_weight, 0, target)
        prob       = (prob*select).sum(1).view(-1,1)
        prob       = torch.clamp(prob,1e-8,1-1e-8)
        batch_loss = - class_weight *(torch.pow((1-prob), self.gamma))*prob.log()
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss
        return loss
```

### Metric

![image.png](https://upload-images.jianshu.io/upload_images/13575947-23578a2dac8e4716.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

项目采用取超过probability超过Thresholds：[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]的IoU均值作为metric。
```
iou_thresholds = np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])

def iou(img_true, img_pred):
    img_pred = (img_pred > 0).float()
    i = (img_true * img_pred).sum()
    u = (img_true + img_pred).sum()
    return i / u if u != 0 else u

def iou_metric(imgs_pred, imgs_true):
    num_images = len(imgs_true)
    scores = np.zeros(num_images)
    for i in range(num_images):
        if imgs_true[i].sum() == imgs_pred[i].sum() == 0:
            scores[i] = 1
        else:
            scores[i] = (iou_thresholds <= iou(imgs_true[i], imgs_pred[i])).mean()
    return scores.mean()
```

### Training

Unet模型训练大致分两步：
- 通过[LR Test](https://arxiv.org/pdf/1506.01186.pdf)找出合适的学习率区间。
- [Cycle Learning Rate (CLR)](https://arxiv.org/pdf/1506.01186.pdf)的方法来训练模型，直至过拟合。
```
wd = 4e-4
arch = resnet34
ps_i = 0.05
ps = np.array([0.1, 0.1, 0.1, 0.1]) * 1
m_base = get_base_model(arch, cut, True)
m = to_gpu(Unet34(m_base, drop_i=True, drop_up=True, ps=ps, ps_i=ps_i))
models = UnetModel(m)
learn = ConvLearner(md, models)
learn.opt_fn = optim.Adam
learn.crit = nn.BCEWithLogitsLoss()
learn.metrics = [accuracy_thresh(0.5), miou]
```
当模型训练到无法通过变化学习率来减少loss值，val loss收敛且有过拟合的可能时，我停止了模型的训练。
![image.png](https://upload-images.jianshu.io/upload_images/13575947-0afe62a185e242e5.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![image.png](https://upload-images.jianshu.io/upload_images/13575947-9a693854572800f8.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
从结果来看模型需要增加正则化来对抗过拟合，Dropout在Unet的实际应用中并没有起到好的效果，所以需要从data augmentation和weight decay下功夫。

### Run length encoder

和rle_decode()相反，在将输出提交到kaggle之前，需要通过rle_encode()根据mask生成相应的run length。当然前提是通过downsample()将mask resize回101x101大小。
```
def downsample(img, shape):
  if shape == img.shape: return img
  return resize(img, shape, mode='constant', preserve_range=True)

def rle_encode(im):
    pixels = im.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)
```

### TTA（Test Time Augmentation）
我们可以通过对testset做data augmentation来提高在kaggle上的score。在segmentation项目中应用TTA时要特别注意的是，augmented images会带来augmented outputs，在对这些outputs求均值之前需要先根据相应的transform规则来转化outputs，例如，image<sub>1</sub>和水平翻转后的image<sub>2</sub>经模型分别生成mark<sub>1</sub>和mark<sub>2</sub>，在计算mark的均值之前需要先将mark<sub>2</sub>做水平翻转。

---

## 小结
到此，Unet模型的构建、训练的几个要点：dataset、model、loss和metric等都已经基本讲清了。这篇博文是我在比赛初期写下的，和我最终使用的模型稍有不同，例如新模型增加了5-folds cross validation、scSE network等， 有时间我会再写篇博文介绍排名靠前的参赛者的方案以及相关技术。


---
