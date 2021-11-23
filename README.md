# Conflict-Averse Gradient Descent for Multitask Learning (CAGrad)
This repo contains the source code for CAGrad, which has been accepted to **NeurIPS 2021**.

## Toy Optimization Problem

![Alt Text](https://github.com/Cranial-XIX/CAGrad/blob/main/misc/cagrad.gif)

To run the toy example:
```
python toy.py
```

## Image-to-Image Prediction
The supervised multitask learning experiments are conducted on NYU-v2 and CityScapes datasets. We follow the setup from [MTAN](https://github.com/lorenmt/mtan). The datasets could be downloaded from [NYU-v2](https://www.dropbox.com/sh/86nssgwm6hm3vkb/AACrnUQ4GxpdrBbLjb6n-mWNa?dl=0) and [CityScapes](https://www.dropbox.com/sh/gaw6vh6qusoyms6/AADwWi0Tp3E3M4B2xzeGlsEna?dl=0). After the datasets are downloaded, please follow the respective run.sh script in each folder. In particular, modify the dataroot variable to the downloaded dataset.

**update 2021/11/22:** Thank [@lushleaf](https://github.com/lushleaf) for finding that replacing the last `backward(retain_graph=True)` to  `backward()` saves much GPU memory.

## Citations
If you find our work interesting or the repo useful, please consider citing [this paper](https://arxiv.org/pdf/2110.14048.pdf):
```
@article{liu2021conflict,
  title={Conflict-Averse Gradient Descent for Multi-task Learning},
  author={Liu, Bo and Liu, Xingchao and Jin, Xiaojie and Stone, Peter and Liu, Qiang},
  journal={arXiv preprint arXiv:2110.14048},
  year={2021}
}
```
