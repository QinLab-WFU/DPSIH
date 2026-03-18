# Polysemic Semantic Instance Network for Cross-Modal Hashing [Paper]( https://ojs.aaai.org/index.php/AAAI/article/view/42459)
This paper is accepted for Proceedings of the 40th Annual AAAI Conference on Artificial Intelligence (AAAI-26)

## Training

### Processing dataset
Refer to [DSPH](https://github.com/QinLab-WFU/DSPH)

### Download CLIP pretrained model
Pretrained model will be found in the 30 lines of [CLIP/clip/clip.py](https://github.com/openai/CLIP/blob/main/clip/clip.py). This code is based on the "ViT-B/32".

You should copy ViT-B-32.pt to this dir.

### Start
> python main.py

## Citation 
If you find this useful for your research, please use the following.

```
@article{Han_Qin_Xie_Zhang_Huang_2026,
title={Polysemic Semantic Instance Network for Cross-Modal Hashing},
author={Han, Shuo and Qin, Qibing and Xie, Kezhen and Zhang, Wenfeng and Huang, Lei},
journal={Proceedings of the AAAI Conference on Artificial Intelligence},
volume={40},
pages={4592-4600},
year={2026}, 
url={https://ojs.aaai.org/index.php/AAAI/article/view/42459},
DOI={10.1609/aaai.v40i6.42459}
}
```

## Acknowledegements
[DCHMT](https://github.com/kalenforn/DCHMT)
