# PSL (Prototypical Similarity Learning)
Official Pytorch Implementation of "[Enlarge Instance-specific and Class-specific Information for Open-set Action Recognition](https://arxiv.org/abs/2303.15467)",

[Jun CEN](www.cen-jun.com), Shiwei Zhang, Xiang Wang, Yixuan Pei, Zhiwu Qing, Yingya Zhang, [Qifeng Chen](https://cqf.io/). In CVPR 2023.

## Table of Contents
1. [Introduction](#introduction)
1. [Installation](#installation)
1. [Datasets](#datasets)
1. [Testing](#testing)
1. [Training](#training)
1. [Model Zoo](#model-zoo)
1. [Citation](#citation)

## Introduction
Open-set action recognition is to reject unknown human action cases which are out of the distribution of the training set. 
Existing methods mainly focus on learning better uncertainty scores but dismiss the importance of the feature representations.
We find that features with richer semantic diversity can significantly improve the open-set performance under the same uncertainty scores.
In this paper, we begin with analyzing the feature representation behavior in the open-set action recognition (OSAR) problem based on the information bottleneck (IB) theory, and propose to enlarge the instance-specific (IS) and class-specific (CS) information contained in the feature for better performance.
For this reason, a novel Prototypical Similarity Learning (PSL) framework is proposed to keep the instance variance within the same class to retain more IS information.
Besides, we notice that unknown samples sharing similar appearances to known samples are easily misclassified as known classes.
To alleviate this issue, video shuffling is further introduced in our PSL to learn distinct temporal information between the original and shuffled samples, which we find enlarges the CS information.
Extensive experiments demonstrate that the proposed PSL can significantly boost both the open-set and closed-set performance and achieves state-of-the-art results on multiple benchmarks. 

## Installation
This repo is developed from [Alibaba-mmai-research](https://github.com/alibaba-mmai-research) codebase.

### Requirements and Dependencies
- Python>=3.6
- torch>=1.5
- torchvision (version corresponding with torch)
- simplejson==3.11.1
- decord>=0.6.0
- pyyaml
- einops
- oss2
- psutil
- tqdm
- pandas

## Datasets

We follow the datasets setting with [DEAR](https://github.com/Cogito2012/DEAR). UCF-101 is for closed-set training, and HMDB-51 and MiT-v2 are for open-set testing.

## Testing

Please refer to `./evaluation_code/README.md` for instructions.

## Training

We provide the training code with TSM backbone as an example.
### Training from scratch
```shell
python tools/run_net.py --cfg configs/projects/openset/tsm/tsm_psl.yaml
```
### Training from K400 pre-trained model
The K400 pre-trained model is [here](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_dense_256p_1x1x8_100e_kinetics400_rgb/tsm_r50_dense_256p_1x1x8_100e_kinetics400_rgb_20200727-e1e0c785.pth).
```shell
python tools/run_net.py --cfg configs/projects/openset/tsm/tsm_psl_ft.yaml
```
We use 8 V100 (32G) or 4 A100 (80G) with batch size 128 for training. The  training consumes around 16 hours.

After training, we store all data required for open-set testing in `output/test/tsm_maha_distance.npz`. Then run
```shell
cd tools/scripts
./compute_openness.sh
```
You should get the closed-set accuracy, open-set AUROC, AUPR, and FPR95.
## Model Zoo

We provide the pre-trained weights (checkpoints) of TSM.
| Model | Checkpoint | Train Config | Closed Set ACC (%) | AUROC (%) | AUPR (%) | FPR95 (%) |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
|PSL |[ckpt](https://drive.google.com/file/d/1h18953Q-VW2pdFXfAYhLSo4pO7mebfvL/view?usp=share_link)| [train](configs/projects/openset/tsm/tsm_psl.yaml) | 77.38 | 86.10 | 64.66 | 43.68 |
|PSL+K400 pretrain|[ckpt](https://drive.google.com/file/d/1oIT1uJQvpUSmkxVlciuL7npkMexkmXei/view?usp=share_link)| [train](configs/projects/openset/tsm/tsm_psl_ft.yaml) | 96.04 | 93.39 | 85.51 | 26.96 |


## Citation
```
@inproceedings{
jun2023enlarge
title={Enlarge Instance-specific and Class-specific Information for Open-set Action Recognition},
author={Jun Cen,  Shiwei Zhang, Xiang Wang, Yixuan Pei, Zhiwu Qing, Yingya Zhang, Qifeng Chen},
booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
year={2023},
}
```

## License

See [Apache-2.0 License](/LICENSE)

## Acknowledgement

In addition to the [Alibaba-mmai-research](https://github.com/alibaba-mmai-research) codebase, this repo contains modified codes from:
 - [DEAR](https://github.com/Cogito2012/DEAR) for implementation of the [DEAR (ICCV-2021)](https://arxiv.org/abs/2107.10161)
 - [pytorch-classification-uncertainty](https://github.com/dougbrion/pytorch-classification-uncertainty): for implementation of the [EDL (NeurIPS-2018)](https://arxiv.org/abs/1806.01768).
 - [ARPL](https://github.com/iCGY96/ARPL): for implementation of baseline method [RPL (ECCV-2020)](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123480511.pdf).
 - [OSDN](https://github.com/abhijitbendale/OSDN): for implementation of baseline method [OpenMax (CVPR-2016)](https://vast.uccs.edu/~abendale/papers/0348.pdf).
 - [bayes-by-backprop](https://github.com/nitarshan/bayes-by-backprop/blob/master/Weight%20Uncertainty%20in%20Neural%20Networks.ipynb): for implementation of the baseline method Bayesian Neural Networks (BNNs).
 - [rebias](https://github.com/clovaai/rebias): for implementation of HSIC regularizer used in [ReBias (ICML-2020)](https://arxiv.org/abs/1910.02806)

We sincerely thank the owners of all these great repos!
