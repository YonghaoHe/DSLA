# DSLA: Dynamic Smooth Label Assignment for Efficient Anchor-free Object Detection
TBD
## Abstract
Anchor-free detectors basically formulate object detection as dense classification and regression. 
For popular anchor-free detectors, it is common to introduce an individual prediction branch to 
estimate the quality of localization. The following inconsistencies are observed when we delve 
into the practices of classification and quality estimation. Firstly, for some adjacent samples 
which are assigned completely different labels, the trained model would produce similar 
classification scores. This violates the training objective and leads to performance degradation. 
Secondly, it is found that detected bounding boxes with higher confidences contrarily have smaller 
overlaps with the corresponding ground-truth. Accurately localized bounding boxes would be 
suppressed by less accurate ones in the Non-Maximum Suppression (NMS) procedure. To address 
the inconsistency problems, the Dynamic Smooth Label Assignment (DSLA) method is proposed. 
Based on the concept of centerness originally developed in FCOS, a smooth assignment strategy 
is proposed. The label is smoothed to a continuous value in [0, 1] to make a steady transition 
between positive and negative samples. Intersection-of-Union (IoU) is predicted dynamically 
during training and is coupled with the smoothed label. The dynamic smooth label is assigned 
to supervise the classification branch. Under such supervision, quality estimation branch is 
naturally merged into the classification branch, which simplifies the architecture of anchor-free 
detector. Comprehensive experiments are conducted on the MS COCO benchmark. It is demonstrated that,
DSLA can significantly boost the detection accuracy by alleviating the above inconsistencies for 
anchor-free detectors. Our paper is available [here](https://arxiv.org/abs/2208.00817).

![Approach](../../resources/img.png)

We adopt the similar network with FCOS except that the centerness branch is
omitted. In the head of our method, only two branches, classification branch 
and regression branch, are reserved. The inner structures are the same as ones 
in FCOS. H and W are height and width of feature maps, C is the number of 
classes. IoU scores are produced by computing the IoU between the predicted 
boxes and the gt boxes.


## Citation
```
@article{DSLA,
  title={DSLA: Dynamic smooth label assignment for efficient anchor-free object detection},
  author={Su, Hu and He, Yonghao and Jiang, Rui and Zhang, Jiabin and Zou, Wei and Fan, Bin},
  journal={Pattern Recognition},
  vol={131},
  pages={108868},
  year={2022}
}
```


## Results and Models

| Backbone  | Style   | GN      | MS train | Tricks  | DCN     | Lr schd | box AP | Config | Download |
|:---------:|:-------:|:-------:|:--------:|:-------:|:-------:|:-------:|:------:|:------:|:--------:|
| R-50      | caffe   | Y       | N        | N       | N       | 1x      | 0.381 | [config](dsla_r50_caffe_fpn_gn-head_1x_coco.py) | [model](https://pan.baidu.com/s/1fi-8CU82REpVfD-i9qexrA) (Pwd:utia) &#124; [log](dsla_r50_20210826_135127.log) |
| R-101     | caffe   | Y       | Y        | Y       | Y       | 2x      |  | [config]() | [model]() &#124; [log]() |
| Swin-S    | -       | Y       | Y        | Y       | Y       | 3x      | 0.492 | [config](dsla_swin-s-p4-w7_fpn_gn-head_3x_coco.py) | [model](https://pan.baidu.com/s/1Nnx1kbURISZAveBDlCU3ow) (Pwd:ggsk) &#124; [log](dsla_swin-s-p4-w7_20220515_200633.log) |