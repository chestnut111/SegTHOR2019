# segthor for keras

 >比赛网址 for more information： https://competitions.codalab.org/competitions/21012  
 
 ![Tumor Segmentation Example](doc/1553665375045.gif)

## label
标签| value | nickname
:--:|:--:|:--:
食管 | 1 | red
心脏 | 2 | pink 
气管 | 3 | yellow
动脉| 4 | blue

## 最后一次的提交结果：

时间 | 食管 | 心脏 | 气管 | 主动脉  
:---:|:---:|:---:|:---:|:---:
3月17日 | 0.8452（第五名）|0.9328（第十八）|0.9039（第十四）|0.9115（第十六）|  
3月22日 | 0.8513（第七名）|0.9457（第九名）|0.9083（第十五）|0.9175（第二十四）|

--   
目前有优于以上的成绩的模型。

## 各个任务的相关实验数据 ：  
食管：  [RED](http://gitlab.ai.chuangxin.com/wanglixin/segthor/wikis/%E5%99%A8%E5%AE%98/RED )   
心脏：  [PINK](http://gitlab.ai.chuangxin.com/wanglixin/segthor/wikis/%E7%8E%8B%E7%AB%8B%E6%96%B0/pink  )  
气管：  [YELLOW](http://gitlab.ai.chuangxin.com/wanglixin/segthor/wikis/%E9%83%91%E5%B0%91%E6%9D%B0/yellow  )  
主动脉： [BLUE](http://gitlab.ai.chuangxin.com/wanglixin/segthor/wikis/%E7%A7%A6%E6%99%8B/blue  )  
  



# Data understanding

## Example Pic
<p align="center">
    <img src="https://docs.google.com/uc?id=1k4ieAL67lmxJhclGGda362BPz982XgVN" alt="Sample"  width="800" >
    <p align="center">
        <em>example</em>
    </p>
</p>

## Training & Testing Data
The whole SegTHOR dataset (60 patients and 11084 slices) has been randomly split into:

- a training set: 40 patients, 7390 slices
- a testing set: 20 patients, 3694 slices

```mermaid
graph LR;
A[train set 40]-->B[for val 8];
A-->C[for train 32];
```

# Train
## Seg  and cls mdoel
Inference  https://github.com/qubvel/segmentation_models

### Avaliable backbones:
| Backbone model      |Name| Weights    |
|---------------------|:--:|:------------:|
| VGG16               |`vgg16`| `imagenet` |
| VGG19               |`vgg19`| `imagenet` |
| ResNet18            |`resnet18`| `imagenet` |
| ResNet34            |`resnet34`| `imagenet` |
| ResNet50            |`resnet50`| `imagenet`<br>`imagenet11k-places365ch` |
| ResNet101           |`resnet101`| `imagenet` |
| ResNet152           |`resnet152`| `imagenet`<br>`imagenet11k` |
| ResNeXt50           |`resnext50`| `imagenet` |
| ResNeXt101          |`resnext101`| `imagenet` |
| DenseNet121         |`densenet121`| `imagenet` |
| DenseNet169         |`densenet169`| `imagenet` |
| DenseNet201         |`densenet201`| `imagenet` |
| Inception V3        |`inceptionv3`| `imagenet` |
| Inception ResNet V2 |`inceptionresnetv2`| `imagenet` |

## Data processing

### Data auto crop 

```mermaid
graph LR;
F[3D cube]-->E(确定图像中心);
E-->V(3D crop)
V-.flatten.->G[单张slice]
```
### Data normalization for CT 

```mermaid
graph LR;
F[train CT image]-->E[HU mean for ev Pat];
L[train mask]-->E[ww and wl];
E-->G(train for norm CT image)
H[test  CT image]-->Q[ww and wl];
O[test  mask predict by older model]-->Q[ww and wl];
Q-->I(test for norm CT image)
```
效果图：
<p align="center">
    <img src="doc/WX20190328-103045@2x.png" alt="Sample"  width="400" >
    <img src="doc/WX20190328-103149@2x.png" alt="Sample"  width="400" >
    <p align="center">
        <em>example</em>
    </p>
</p>

