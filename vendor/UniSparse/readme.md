# UniSparse: Combining Weight Pruning and Spike Sparsification in Spiking Neural Networks

## Installing Dependencies

```bash
pip3 install torch torchvision
pip3 install spikingjelly==0.0.0.0.14 cupy-cuda12x timm tensorboard thop
```

## Usage

Use the commands below to reproduce the experiments described in the paper.

### CIFAR-10

```bash
python main.py -c configs/cifar10_cifarnet.yaml --lmbda-weight <lmbda> --lmbda-neuron <lmbda>
```

```bash
python main.py -c configs/cifar10_res19.yaml --lmbda-weight <lmbda> --lmbda-neuron <lmbda>
```

```bash
python main.py -c configs/cifar10_res18.yaml --lmbda-weight <lmbda> --lmbda-neuron <lmbda>
```

### DVS-CIFAR10

```bash
python main.py -c configs/cifar10dvs_vggsnn.yaml --lmbda-weight <lmbda> --lmbda-neuron <lmbda>
```

### ImageNet

To reproduce the experiments on ImageNet, first download the ImageNet dataset manually from [here](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php).

Place the training images in the `train` directory and the validation images in the `val` directory. Then run:

```bash
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=4 \
main.py \
    -c configs/imagenet_res18.yaml \
    --data-path /path/to/your/dataset \
    --lmbda-weight <lmbda> \
    --lmbda-neuron <lmbda> \
    ;
```

## Fix bug

`spikingjelly` version 0.0.0.0.14 is incompatible with `numpy` 1.20 or later. To resolve this, manually edit the file located at `/path/to/your/spikingjelly/activation_based/auto_cuda/base.py`, line 249:

```python
    elif value.dtype == np.int:
        assert startswiths(ctype, ('const int', 'int'))
```

Change `np.int` to `np.int_`:

```python
    elif value.dtype == np.int_:
        assert startswiths(ctype, ('const int', 'int'))
```

## Citation

If you find this code useful for your research, please consider citing our paper:

```bibtex
@inproceedings{shi2026unisparse,
  title={UniSparse: Combining Weight Pruning and Spike Sparsification in Spiking Neural Networks},
  author={Shi, Xinyu and Bu, Tong and Yu, Zhaofei},
  booktitle={International Conference on Machine Learning},
  year={2026}
}
```
