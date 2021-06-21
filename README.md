# co-mod-gan-pytorch
Implementation of the paper ``Large Scale Image Completion via Co-Modulated Generative Adversarial Networks"

official tensorflow version: https://github.com/zsyzzsoft/co-mod-gan

Input image<img src="imgs/example_image.jpg" width=200> Mask<img src="imgs/example_mask.jpg" width=200>  Result<img src="imgs/example_output.jpg" width=200>  

## Usage

### requirments
```
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.2 -c pytorch
conda install pillow
```

### inference 

1. download pretrained model using ``download_*.sh" (converted from the tensorflow pretrained model)

e.g. ffhq512

```
./download_ffhq512.sh
```

2. use the following command as a minimal example of usage

```
python test.py -i imgs/example_image.jpg -m imgs/example_mask.jpg -o ./imgs/example_output.jpg -c checkpoints/co-mod-gan-ffhq-9-025000.pth
```

### Training
Coming soon

## Reference

[1] official tensorflow version: https://github.com/zsyzzsoft/co-mod-gan

[2] stylegan2-pytorch https://github.com/rosinality/stylegan2-pytorch
