# OrthoMAD: Morphing Attack Detection Through Orthogonal Identity Disentanglement
Official repository for the OrthoMAD: Morphing Attack Detection Through Orthogonal Identity Disentanglement paper at [BIOSIG 2022](https://biosig.de).

The paper can be viewed at: [proceedings](https://ieeexplore.ieee.org/document/9897057) or [arXiv](https://arxiv.org/abs/2208.07841)

## Abstract
Morphing attacks are one of the many threats that are constantly affecting deep face recognition systems. It consists of selecting two faces from different individuals and fusing them into a final image that contains the identity information of both. In this work, we propose a novel regularisation term that takes into account the existent identity information in both and promotes the creation of two orthogonal latent vectors. We evaluate our proposed method (OrthoMAD) in five different types of morphing in the FRLL dataset and evaluate the performance of our model when trained on five distinct datasets. With a small ResNet-18 as the backbone, we achieve state-of-the-art results in the majority of the experiments, and competitive results in the others.

## How to run

Example command: 
```bash
python3 code/train.py --train_csv_path="morgan_lma_train.csv" --test_csv_path="morgan_test.csv" --max_epoch=250 --batch_size=16 --latent_size=32 --lr=0.00001 --weight_loss=100
```

## Acknowledgement
The code was extended from the initial code of [SMDD-Synthetic-Face-Morphing-Attack-Detection-Development](https://github.com/naserdamer/SMDD-Synthetic-Face-Morphing-Attack-Detection-Development-dataset). 

## Citation
If you use our code or data in your research, please cite with:

```bibtex
@INPROCEEDINGS{neto2022orthomad,
  author={Neto, Pedro C. and Gonçalves, Tiago and Huber, Marco and Damer, Naser and Sequeira, Ana F. and Cardoso, Jaime S.},
  booktitle={2022 International Conference of the Biometrics Special Interest Group (BIOSIG)}, 
  title={OrthoMAD: Morphing Attack Detection Through Orthogonal Identity Disentanglement}, 
  year={2022},
  volume={},
  number={},
  pages={1-5},
  doi={10.1109/BIOSIG55365.2022.9897057}}
```
