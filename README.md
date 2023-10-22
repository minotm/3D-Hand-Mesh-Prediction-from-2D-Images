[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen)](https://opensource.org/licenses/MIT)

# 3D Interacting Hand Mesh Prediction from 2D RGB Images

2022 course project

3D hand surface reconstruction from monocular images has a wide range of potential use cases including AR and robotics. Previous efforts have focused on inputs with a single hand. In the wild, however, both hands frequently interact with one another and perception techniques can be confounded due to the occurrence of occlusion or highly interacting hand poses. To this end, leveraging a portion of the InterHand2.6M dataset and components and concepts from the MANO, HMR, and an Interacting Attention Graph model, we attempt to predict 3D hand surfaces from 2D images.  

Additional detail can be found in `summary_report.pdf` 

Example usage: `python scripts/train.py --num_epoch=45 --eval_every_epoch=10 --trainsplit=train --valsplit=val --num_workers=6`


## Citations
InterHand2.6M: A Dataset and Baseline for 3D Interacting Hand Pose Estimation from a Single RGB Image [[GitHub](https://mks0601.github.io/InterHand2.6M/)]:

```
@misc{moon2020interhand26m,
      title={InterHand2.6M: A Dataset and Baseline for 3D Interacting Hand Pose Estimation from a Single RGB Image}, 
      author={Gyeongsik Moon and Shoou-i Yu and He Wen and Takaaki Shiratori and Kyoung Mu Lee},
      year={2020},
      eprint={2008.09309},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

Interacting Attention Graph for Single Image Two-Hand Reconstruction [[GitHub](https://github.com/Dw1010/IntagHand)]:

```
@misc{li2022interacting,
      title={Interacting Attention Graph for Single Image Two-Hand Reconstruction}, 
      author={Mengcheng Li and Liang An and Hongwen Zhang and Lianpeng Wu and Feng Chen and Tao Yu and Yebin Liu},
      year={2022},
      eprint={2203.09364},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

