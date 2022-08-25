# OrthoMAD
Official repository for the OrthoMAD: Morphing Attack Detection Through Orthogonal Identity Disentanglement paper at BIOSIG 2022.

The paper can be viewed at: [arXiv](https://arxiv.org/abs/2208.07841)

## How to run

Example command: 
```
python3 train.py --train_csv_path="morgan_lma_train.csv" --test_csv_path="morgan_test.csv" --max_epoch=250 --batch_size=16 --latent_size=32 --lr=0.00001 --weight_loss=100
```

## Acknowledgement
The code was extended from the initial code of [SMDD-Synthetic-Face-Morphing-Attack-Detection-Development](https://github.com/naserdamer/SMDD-Synthetic-Face-Morphing-Attack-Detection-Development-dataset). 

## Citation
If you use our code or data in your research, please cite with:

```
@article{neto2022orthomad,
  title={OrthoMAD: Morphing Attack Detection Through Orthogonal Identity Disentanglement},
  author={Neto, Pedro C and Gon{\c{c}}alves, Tiago and Huber, Marco and Damer, Naser and Sequeira, Ana F and Cardoso, Jaime S},
  journal={arXiv preprint arXiv:2208.07841},
  year={2022}
}
```
