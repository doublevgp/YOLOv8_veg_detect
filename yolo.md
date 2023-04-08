# YOLOv8

在 YOLOv8 目标检测中增加特定类别的权重，可以通过调整损失函数中不同类别的权重来实现。具体来说，可以使用交叉熵损失函数，对每个类别的预测结果进行加权。

## 交叉熵损失函数

在 YOLOv8 中，交叉熵损失函数通常表示为：

$L_{cls} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{j=1}^{S^2}\sum_{c=1}^{C}[y_{ij}^c\log(\hat{y}{ij}^c)+(1-y{ij}^c)\log(1-\hat{y}_{ij}^c)]$

其中，$N$表示训练集的样本数量，$S$ 表示特征图的大小，$C$ 表示类别数量。$y_{ij}^c$ 表示第 $i$ 个样本在特征图中的第 $j$ 个位置预测的类别是否为第 $c$ 个类别，$\hat{y}_{ij}^c$ 表示模型预测第 $i$ 个样本在特征图中的第 $j$ 个位置为第 $c$ 个类别的概率值。

为了增加特定类别的权重，我们可以对交叉熵损失函数中特定类别的权重进行增加，例如对第 $c$个类别的权重增加一个系数 $\alpha$，得到新的交叉熵损失函数：

$L_{cls} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{j=1}^{S^2}\sum_{c=1}^{C}[y_{ij}^c\alpha\log(\hat{y}{ij}^c)+(1-y{ij}^c)\log(1-\hat{y}_{ij}^c)]$

其中，$\alpha$ 是一个大于1的系数，可以根据需要进行调整。增加特定类别的权重可以使模型更加关注这些类别，在训练过程中更加注重这些类别的表现。

在 YOLOv8 的实现中，可以通过调整配置文件中的 cls_normalizer 参数来实现增加特定类别的权重。例如，将 cls_normalizer 设置为 [1.0, 1.0, 1.0, 10.0, 1.0, 1.0]，可以将第4个类别的权重增加到其他类别的10倍。

需要注意的是，在增加特定类别的权重时，需要确保该类别的样本数量充足，否则可能会导致过拟合。另外，如果训练集中存在类别不平衡问题，可以使用其他方法来调整类别权重，如 Focal Loss 等。

## Early Stopping

早停法，防止模型在训练过程中过早拟合，当模型在验证集上的表现开始下降时，停止训练，避免训练导致过拟合

![img](https://img-blog.csdnimg.cn/20191204161025655.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MDQ0NjU1Nw==,size_16,color_FFFFFF,t_70)

可能出现的原因

1. 学习率设置过高：学习率过高可能导致训练过程不稳定，出现早停现象。可以尝试减小学习率或者使用自适应学习率算法，如 Adam、Adagrad、RMSprop 等。
2. 模型结构不合理：模型结构可能过于简单或复杂，无法有效地学习数据的特征。可以尝试调整模型结构或者使用预训练模型进行迁移学习。
3. 数据集不足或不平衡：如果您的数据集过小或者类别不平衡，可能会导致模型在某些类别上表现不佳。可以尝试增加数据集的大小、使用数据增强技术或调整类别权重。
4. 训练过程中出现梯度消失或爆炸：梯度消失或爆炸可能导致模型无法继续优化，出现早停现象。可以尝试使用更合适的激活函数、权重初始化方法或者梯度裁剪技术。
5. 超参数设置不合理：模型的超参数设置可能不合适，导致模型在训练过程中出现早停现象。可以尝试调整超参数，如批量大小、正则化系数等。

调整措施

1. 增加训练数据量：增加数据量可以提高模型的泛化能力，减少出现早停的可能性。
2. 调整模型结构：根据具体情况调整模型结构，例如增加网络深度、加入更多的卷积层或者使用残差结构等。
3. 调整超参数：根据具体情况调整超参数，如学习率、批量大小、正则化系数等。
4. 使用预训练模型或迁移学习：预训练模型或迁移学习可以利用已有的模型结构和参数，在有限的数据集上进行训练，提高模型的泛化能力。
5. 调整类别权重：如果数据集中存在类别不平衡问题，可以尝试调整类别权重，以提高模型在少数类别上的表现。



## YOLOv8 Train Arguments

Training settings for YOLO models refer to the various hyperparameters and configurations used to train the model on a dataset. These settings can affect the model's performance, speed, and accuracy. Some common YOLO training settings include the batch size, learning rate, momentum, and weight decay. Other factors that may affect the training process include the choice of optimizer, the choice of loss function, and the size and composition of the training dataset. It is important to carefully tune and experiment with these settings to achieve the best possible performance for a given task.

| Key               | Value    | Description                                                  | 理解                                                         |
| ----------------- | -------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `model`           | `None`   | path to model file, i.e. yolov8n.pt, yolov8n.yaml            | 权重模型                                                     |
| `data`            | `None`   | path to data file, i.e. coco128.yaml                         | 数据集                                                       |
| `epochs`          | `100`    | number of epochs to train for                                | 训练轮数                                                     |
| `patience`        | `50`     | epochs to wait for no observable improvement for early stopping of training | 早停轮数                                                     |
| `batch`           | `16`     | number of images per batch (-1 for AutoBatch)                | 批处理量                                                     |
| `imgsz`           | `640`    | size of input images as integer or w,h                       | 文件大小                                                     |
| `save`            | `True`   | save train checkpoints and predict results                   | 保存结果                                                     |
| `save_period`     | `-1`     | Save checkpoint every x epochs (disabled if < 1)             |                                                              |
| `cache`           | `False`  | True/ram, disk or False. Use cache for data loading          |                                                              |
| `device`          | `None`   | device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu | cpu训练还是gpu                                               |
| `workers`         | `8`      | number of worker threads for data loading (per RANK if DDP)  | 多线程工作                                                   |
| `project`         | `None`   | project name                                                 | /                                                            |
| `name`            | `None`   | experiment name                                              | /                                                            |
| `exist_ok`        | `False`  | whether to overwrite existing experiment                     | 文件夹名称是否可重复                                         |
| `pretrained`      | `False`  | whether to use a pretrained model                            | 预训练                                                       |
| `optimizer`       | `'SGD'`  | optimizer to use, choices=['SGD', 'Adam', 'AdamW', 'RMSProp'] |                                                              |
| `verbose`         | `False`  | whether to print verbose output                              |                                                              |
| `seed`            | `0`      | random seed for reproducibility                              | 随机数通常被用来进行数据增强、权重初始化、随机采样等操作，如果需要训练过程可复现，每次需要指定相同的```seed``` |
| `deterministic`   | `True`   | whether to enable deterministic mode                         | 确定性运算模式，相同的输入，结果相同                         |
| `single_cls`      | `False`  | train multi-class data as single-class                       | 以单类模型训练多类                                           |
| `image_weights`   | `False`  | use weighted image selection for training                    | 图像权重，可以给图像设置不同的权重```image_weights <weights_file>```  `<weights_file>` 是包含图像权重列表的文本文件。在文本文件中，每行包含一个图像的权重值，按照训练集中的顺序排列。 |
| `rect`            | `False`  | rectangular training with each batch collated for minimum padding | 矩形推理                                                     |
| `cos_lr`          | `False`  | use cosine learning rate scheduler                           | 余弦学习率调度器，学习率高，模型收敛快，学习率低，模型收敛稳定 |
| `close_mosaic`    | `10`     | disable mosaic augmentation for final 10 epochs              | 在最后10轮的时候关闭马赛克数据增强算法                       |
| `resume`          | `False`  | resume training from last checkpoint                         | 继续训练                                                     |
| `amp`             | `True`   | Automatic Mixed Precision (AMP) training, choices=[True, False] |                                                              |
| `lr0`             | `0.01`   | initial learning rate (i.e. SGD=1E-2, Adam=1E-3)             | 初始学习率                                                   |
| `lrf`             | `0.01`   | final learning rate (lr0 * lrf)                              | 最终学习率                                                   |
| `momentum`        | `0.937`  | SGD momentum/Adam beta1                                      | 权重更新的动量大小                                           |
| `weight_decay`    | `0.0005` | optimizer weight decay 5e-4                                  |                                                              |
| `warmup_epochs`   | `3.0`    | warmup epochs (fractions ok)                                 |                                                              |
| `warmup_momentum` | `0.8`    | warmup initial momentum                                      |                                                              |
| `warmup_bias_lr`  | `0.1`    | warmup initial bias lr                                       |                                                              |
| `box`             | `7.5`    | box loss gain                                                |                                                              |
| `cls`             | `0.5`    | cls loss gain (scale with pixels)                            |                                                              |
| `dfl`             | `1.5`    | dfl loss gain                                                |                                                              |
| `pose`            | `12.0`   | pose loss gain (pose-only)                                   |                                                              |
| `kobj`            | `2.0`    | keypoint obj loss gain (pose-only)                           |                                                              |
| `fl_gamma`        | `0.0`    | focal loss gamma (efficientDet default gamma=1.5)            |                                                              |
| `label_smoothing` | `0.0`    | label smoothing (fraction)                                   |                                                              |
| `nbs`             | `64`     | nominal batch size                                           |                                                              |
| `overlap_mask`    | `True`   | masks should overlap during training (segment train only)    |                                                              |
| `mask_ratio`      | `4`      | mask downsample ratio (segment train only)                   | 随机地将输入图像中的一部分区域进行遮挡，然后再将遮挡后的图像作为输入图像进行训练 |
| `dropout`         | `0.0`    | use dropout regularization (classify train only)             | 随机地将一部分神经元的输出置为零，从而使得模型的每个部分都能够独立地学习和适应数据。这样可以减少模型中神经元之间的依赖关系，提高模型的泛化能力和鲁棒性，从而避免模型在训练集上表现良好但在测试集上表现不佳的情况 |
| `val`             | `True`   | validate/test during training                                | 在训练中是否使用验证集验证模型                               |

详见[YOLOv8文档](https://docs.ultralytics.com/)

