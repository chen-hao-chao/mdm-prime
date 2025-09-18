<div align="center">
<br>
<img src="assets/title.png" width="500">
</div>

<br>
<p align="center">
<a href="https://arxiv.org/abs/2505.18495"><img src="https://img.shields.io/badge/arXiv-2505.18495-b31b1b.svg?logo=arxiv&logoColor=red" alt="MDM-Prime Paper on arXiv"/></a>
<a href="https://huggingface.co/chen-hao-chao/mdm-prime"><img src="https://img.shields.io/badge/🤗_HuggingFace%20-MDM_Prime%20-orange" alt="MDM-Prime on Hugging Face"/></a>
<a href="https://hub.docker.com/r/chenhaochao/mdm-prime"><img src="https://img.shields.io/badge/dockerhub-MDM_Prime-blue.svg?logo=docker" alt="MDM-Prime on Docker"/></a>
<a href="https://hub.docker.com/r/chenhaochao/mdlm-prime"><img src="https://img.shields.io/badge/dockerhub-MDLM_Prime-blue.svg?logo=docker" alt="MDLM-Prime on Docker"/></a><br>
<!-- <a href="https://paperswithcode.com/sota/language-modelling-on-openwebtext?p=beyond-masked-and-unmasked-discrete-diffusion"><img src="https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/beyond-masked-and-unmasked-discrete-diffusion/language-modelling-on-openwebtext" alt="MDM-Prime on Paper with Code"/></a><br>
<a href="https://paperswithcode.com/sota/image-generation-on-cifar-10?p=beyond-masked-and-unmasked-discrete-diffusion"><img src="https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/beyond-masked-and-unmasked-discrete-diffusion/image-generation-on-cifar-10" alt="MDM-Prime on Paper with Code"/></a><br>
<a href="https://paperswithcode.com/sota/image-generation-on-imagenet-32x32?p=beyond-masked-and-unmasked-discrete-diffusion"><img src="https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/beyond-masked-and-unmasked-discrete-diffusion/image-generation-on-imagenet-32x32" alt="MDM-Prime on Paper with Code"/></a>
</p> -->

## What’s Inside

This repository contains the code implementation of the experiments presented in the paper [*Beyond Masked and Unmasked: Discrete Diffusion Models via Partial Masking*](https://arxiv.org/abs/2505.18495).

- :whale: **Docker environments** for enhanced reproducibility
- :cd: **Pretrained weights** for inference and evaluation
- :microscope: **Code for all experiments** in our paper:
  - Toy experiments on synthetic data
  - Text generation on OpenWebText
  - Image generation on CIFAR-10 & ImageNet-32

## Overview

### Toy Examples

- **Dataset**: 2D Synthetic Dataset  
- **Folder**: [mdm-prime/toy](/toy)
- <details> <summary> <strong>Demo</strong> (click me) </summary> <img src="toy/assets/toy_demo.png" alt="prime_toy" width="600px"> </details>

### Text Generation
- **Dataset**: OpenWebText (OWT)
- **Folder**: [mdm-prime/text](/text)
- <details> <summary> <strong>Demo</strong> (click me) </summary> <img src="text/assets/text_demo.gif" alt="prime_text" width="800px"> </details>


### Image Generation
- **Dataset**: CIFAR-10, ImageNet-32
- **Folder**: [mdm-prime/image](/image)
- <details> <summary> <strong>Demo</strong> (click me) </summary> <img src="image/assets/img_demo.gif" alt="prime_img" width="800px"> </details>

## License
This code implementation is developed based on the following repositories.

- [kuleshov-group/mdlm](https://github.com/kuleshov-group/mdlm) (at commit `3ecb6dc`), licensed under the `Apache-2.0` license.
- [facebookresearch/flow_matching](https://github.com/facebookresearch/flow_matching) (at commit `c056dd6`), licensed under the `CC BY-NC 4.0` license.

Further changes based on this repository are licensed under the `Apache-2.0` and `CC BY-NC 4.0` licenses.


## Citing MDM-Prime

If you find this code implementation useful, please consider citing our paper.

```bib
@inproceedings{chao2025mdmprime,
      title = {{Beyond Masked and Unmasked: Discrete Diffusion Models via Partial Masking}}, 
      author = {Chen-Hao Chao, Wei-Fang Sun, Hanwen Liang, Chun-Yi Lee, Rahul G. Krishnan},
      booktitle = {Proceedings of the Conference on Neural Information Processing Systems (NeurIPS)},
      year = {2025},
}
```
