# Two-Dimensional Toy Examples

[![arXiv](https://img.shields.io/badge/arXiv-2505.18495-b31b1b.svg?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2505.18495) [<img src="https://img.shields.io/badge/dockerhub-MDM_Prime-blue.svg?logo=docker">](https://hub.docker.com/r/chenhaochao/mdm-prime)<br>

This folder contains the code implementation of the toy experiments presented in **Fig. 4** of [our paper](https://arxiv.org/abs/2505.18495).

<img src="assets/toy_demo.png" alt="toy_demo" width="600px">

> Images are from the [Cat dataset](https://www.kaggle.com/datasets/crawford/cat-dataset). Distributions are sampled using MDM-Prime with $\ell=2$.

## Install Dependencies

You can choose to lunch our pre-built docker image or manually install the dependencies via conda:

### :whale: Docker

1. Pull our pre-built docker image:
```bash
docker pull chenhaochao/mdm-prime:latest
```
or build a docker image locally through the following command:

```bash
docker build -t mdm-prime:latest .
```

2. Launch the docker image through the following commands:
```bash
# assume the current directory (mdm-prime/text) is the root of this repository
docker run -v $(pwd):/app --rm -it --gpus all --ipc=host chenhaochao/mdm-prime:latest
# inside the docker container, run:
cd /app
```
    
### :snake: Conda
    
- Install conda environment:

```bash
# Create conda environment
conda env create -f environment.yml
conda activate mdm-prime
# Install default dependencies via pip
pip install -r requirements.txt
```

- **Note**: The toy and image experiments share the same dependencies. You can skip this installation step if the dependencies are already installed.

<details>
<summary><strong>Possible Error Messages & Solutions</strong></summary>

**Error**. When executing `pip install -r requirements.txt`:
```bash
Building wheels for collected packages: LibCST
  Building wheel for LibCST (pyproject.toml) ... error
  error: subprocess-exited-with-error
  × Building wheel for LibCST (pyproject.toml) did not run successfully.
  │ exit code: 1
  ╰─> [435 lines of output]
```
**Solution**. Install `sympy` with the following command and then install `requirements.txt` again:
```
pip install sympy==1.13.1
```

</details>

## Commands

### Training

:pushpin: **MDM-Prime**

```bash
python main.py --target_length 2 \
               --mode train \
               --image_path assets/cat_1.jpg \
               --rootdir /app \
               --workdir results \
               --resume_step 0 
```

- **Arguments:**
    - `--target_length`: number of sub-tokens, i.e., $\ell$. (default: `1`)
    - `--mode`: `train`, `sample`, or `eval_elbo`. (default: `train`)
    - `--image_path`: path to the target image. (default: `assets/cat_1.jpg`)
    - `--rootdir`: path to the root directory. (default: `/app`)
    - `--workdir`: name of the working directory. (default: `results`)
    - `--resume_step`: the training iteration to resume from. (default: `0`)

> One can customize the target distribution by specifying the `--image_path` argument. This repository provides some example images in the `assets` folder, sourced from the [Cat dataset](https://www.kaggle.com/datasets/crawford/cat-dataset). The customized image should have the default resolution of 512×512; alternatively, `vocab_size` should be adjusted to match the image dimensions.

> The model's weights and sampling results will be saved at `${rootdir} / ${workdir} / l=${target_length}_image=${image_file_name}`.

### Inference

:pushpin: **Sampling**

```bash
python main.py --target_length 2 \
               --mode sample \
               --image_path assets/cat_1.jpg \
               --rootdir /app \
               --workdir results \
               --resume_step 1000000 \
               --temperature 1.0 \
               --nfe 128
```

- **Arguments:**
    - `--nfe`: number of function evaluations. (default: `128`)
    - `--temperature`: the parameter for sharpening or smoothing the distribution. (default: `1.0`)
    - `--save_path`: the path where results will be saved at. (default: `./`)

:pushpin: **Evaluating ELBO**

```bash
python main.py --target_length 2 \
               --mode eval_elbo \
               --image_path assets/cat_1.jpg \
               --rootdir /app \
               --workdir results \
               --resume_step 1000000
```


### Prertained Weights

|   | `assets/cat_1.jpg` | `assets/cat_2.jpg` | `assets/cat_3.jpg` |
| - | ------------------ | ------------------ | ------------------ |
| <p align="center">Google Drive Link</p> | <p align="center">[Download Link](https://drive.google.com/file/d/1Z6YrsZZHhKJbbaVGIILL9QLKC3M1Uagx/view?usp=sharing)</p> | <p align="center">[Download Link](https://drive.google.com/file/d/1HESTDJ08jtgM4opO8XZ55k7uAZRKbUWD/view?usp=sharing)</p>| <p align="center">[Download Link](https://drive.google.com/file/d/1qdf8e5cMTlLyUGYLehlT8CoEzOL38O8H/view?usp=sharing)</p> |
| `<file_id>`    | 1Z6YrsZZHhKJbbaVGIILL9QLKC3M1Uagx | 1HESTDJ08jtgM4opO8XZ55k7uAZRKbUWD | 1qdf8e5cMTlLyUGYLehlT8CoEzOL38O8H |

Download the files manually and place these folders at `${rootdir} / ${workdir}`, or download these files using `gdown` using the following commands:
```bash
pip install gdown
gdown <file_id>
```

## License

This code implementation is developed based on the following repository.

- [facebookresearch/flow_matching](https://github.com/facebookresearch/flow_matching) (at commit `c056dd6`), licensed under the `CC BY-NC 4.0` license.

Further changes based on the code in this folder are licensed under the `CC BY-NC 4.0` license.


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
