# Adversarial-Single-Domain-Generalization
This is the official pytorch implementation of "Adversarial Consistency for Single Domain Generalization in Medical Image Segmentation" of MICCAI2022

This repository is by [Yanwu Xu](http://xuyanwu.github.io)
and contains the [PyTorch](https://pytorch.org) source code to
reproduce the experiments in our MICCAI2022 paper [Adversarial Consistency for Single Domain Generalization in Medical Image Segmentation](https://arxiv.org/pdf/2206.13737.pdf) by [Yanwu Xu](http://xuyanwu.github.io), Shaoan Xie, Maxwell Reynolds, Matthew Ragoza, Mingming Gong* and Kayhan Batmanghelich* (* Equal Contribution)

| Model Structure |
:-------------------------:|
![1.0](figures/model_v2.pdf)  |

## Face Pose Transfer data can be downloaded [here](https://drive.google.com/file/d/1AUoRl0_suTIunTTJ5jor8poUmkoKfxMb/view?usp=sharing)

## Experiments on real data

To run the code on the face pose transfer data. 1. download the data from above link 2. unzip the data to the ./data folder 3. sh run_gcpert.sh

## Qualitative Results
| Comparison |
:-------------------------:|
![1.0](figure/qualitative.png)  |

## Dynamic of Spatial Transformer T
| Comparison |
:-------------------------:|
![1.0](figure/face_epoch.png)  |

# Citation

```
@InProceedings{10.1007/978-3-031-16449-1_64,
author="Xu, Yanwu
and Xie, Shaoan
and Reynolds, Maxwell
and Ragoza, Matthew
and Gong, Mingming
and Batmanghelich, Kayhan",
editor="Wang, Linwei
and Dou, Qi
and Fletcher, P. Thomas
and Speidel, Stefanie
and Li, Shuo",
title="Adversarial Consistency for Single Domain Generalization in Medical Image Segmentation",
booktitle="Medical Image Computing and Computer Assisted Intervention -- MICCAI 2022",
year="2022",
publisher="Springer Nature Switzerland",
address="Cham",
pages="671--681",
abstract="An organ segmentation method that can generalize to unseen contrasts and scanner settings can significantly reduce the need for retraining of deep learning models. Domain Generalization (DG) aims to achieve this goal. However, most DG methods for segmentation require training data from multiple domains during training. We propose a novel adversarial domain generalization method for organ segmentation trained on data from a single domain. We synthesize the new domains via learning an adversarial domain synthesizer (ADS) and presume that the synthetic domains cover a large enough area of plausible distributions so that unseen domains can be interpolated from synthetic domains. We propose a mutual information regularizer to enforce the semantic consistency between images from the synthetic domains, which can be estimated by patch-level contrastive learning. We evaluate our method for various organ segmentation for unseen modalities, scanning protocols, and scanner sites.",
isbn="978-3-031-16449-1"
}
```

# Acknowledgments

This work was partially supported by NIH Award Number 1R01HL141813-01, NSF 1839332 Tripod+X, SAP SE, and Pennsylvania Department of Health. We are grateful for the computational resources provided by Pittsburgh SuperComputing grant number TG-ASC170024. MG is supported by Australian Research Council Project DE210101624. KZ would like to acknowledge the support by the National Institutes of Health (NIH) under Contract R01HL159805, by the NSF-Convergence Accelerator Track-D award #2134901, and by the United States Air Force under Contract No. C7715.
