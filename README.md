# co-mod-gan-pytorch
Implementation of the paper ``Large Scale Image Completion via Co-Modulated Generative Adversarial Networks"

official tensorflow version: https://github.com/zsyzzsoft/co-mod-gan

Input image<img src="imgs/ffhq_in.png" width=200> Mask<img src="imgs/ffhq_m.png" width=200>  Result<img src="imgs/example_output.jpg" width=200> 

## Usage

### requirments
```
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.2 -c pytorch
conda install pillow
```

### inference 

1. download pretrained model using ``download/*.sh" (converted from the tensorflow pretrained model)

e.g. ffhq512

```
./download/ffhq512.sh
```

converted model:
* FFHQ 512 checkpoints/comod-ffhq-512/co-mod-gan-ffhq-9-025000_net_G_ema.pth
* FFHQ 1024 checkpoints/comod-ffhq-1024/co-mod-gan-ffhq-10-025000_net_G_ema.pth
* Places 512 checkpoints/comod-places-512/co-mod-gan-places2-050000_net_G_ema.pth

2. use the following command as a minimal example of usage

```
./test.sh
```

### Training
1. download example datasets for training and validation

```
./download/data.sh
```

2. use the following command as a minimal example of usage

```
./train.sh
```

### Demo
Coming soon

## Reference

[1] official tensorflow version: https://github.com/zsyzzsoft/co-mod-gan

[2] stylegan2-pytorch https://github.com/rosinality/stylegan2-pytorch
