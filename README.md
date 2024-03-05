# Discrete-Probability-Flow

The source code for our paper "Tackling the Singularities at the Endpoints of Time Intervals in Diffusion Models", Pengze Zhang*, Hubery Yin*, Chen Li, Xiaohua Xie, CVPR 2024.

<div align=center>
<img width="1148" alt="framework" src="https://github.com/PangzeCheung/SingDiffusion/assets/37894893/76fc771d-ec33-4fb5-ab03-ba5265f31a3b">
</div>

## Abstract

Most diffusion models assume that the reverse process adheres to a Gaussian distribution. However, this approximation has not been rigorously validated, especially at singularities, where t=0 and t=1. Improperly dealing with such singularities leads to an average brightness issue in applications, and limits the generation of images with extreme brightness or darkness. We primarily focus on tackling singularities from both theoretical and practical perspectives. Initially, we establish the error bounds for the reverse process approximation, and showcase its Gaussian characteristics at singularity time steps. Based on this theoretical insight, we confirm the singularity at t=1 is conditionally removable while it at t=0 is an inherent property. Upon these significant conclusions, we propose a novel plug-and-play method SingDiffusion to address the initial singular time step sampling, which not only effectively resolves the average brightness issue for a wide range of diffusion models without extra training efforts, but also enhances their generation capability in achieving notable lower FID scores. Code and models are released.

<div align=center>
<img width="800" alt="framework" src="https://github.com/PangzeCheung/SingDiffusion/assets/37894893/22c69fbf-ea8a-434c-8be5-0dfc27395a14">
</div>

## 1) Get start

* Python 3.9.0
* CUDA 11.2
* NVIDIA A100 40GB PCIe
* Torch 2.0.1
* Torchvision 0.15.2

Please follow **[diffusers](https://github.com/huggingface/diffusers)** to install diffusers.

## 2) Install pre-trained **[SingDiffusion module](https://drive.google.com/drive/folders/1wPZDRPcsnToRobu0ssBEg6cPV2TRGMAi?usp=sharing)** into ./SingDiffusion

## 3) Generate image for testing average brightness issue

**Sampling with SingDiffusion**
```bash
python python test_sing_diffusion_img2img_average_brightness.py --out_dir XXX 
```

**Sampling without SingDiffusion**
```bash
python python test_sing_diffusion_img2img_average_brightness.py --out_dir XXX --no_SingDiffusion
```

## 4) Generate image for testing COCO dataset

Download 30K **[COCO prompt](https://drive.google.com/file/d/1TcYgGyQ2hGRktuBcaeISKKXjTf99cqoW/view?usp=sharing)** into ./COCO_3W_prompt.json

**Sampling with SingDiffusion**
```bash
python python test_sing_diffusion_img2img_COCO.py --out_dir XXX 
```

**Sampling without SingDiffusion**
```bash
python python test_sing_diffusion_img2img_COCO.py --out_dir XXX --no_SingDiffusion
```


## Citation

```tex
@inproceedings{
zhang2024tackling,
title={Tackling the Singularities at the Endpoints of Time Intervals in Diffusion Models},
author={Pengze Zhang and Hubery Yin and Chen Li and Xiaohua Xie},
booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
year={2024}
}
```

## Acknowledgement 

We build our project based on **[diffusers](https://github.com/huggingface/diffusers)**. We thank them for their wonderful work and code release.
