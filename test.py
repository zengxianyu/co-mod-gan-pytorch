import argparse
import numpy as np
import torch
from co_mod_gan import Generator
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--checkpoint', help='Network checkpoint path', required=True)
parser.add_argument('-i', '--image', help='Original image path', required=True)
parser.add_argument('-m', '--mask', help='Mask path', required=True)
parser.add_argument('-o', '--output', help='Output (inpainted) image path', required=True)
parser.add_argument('-t', '--truncation', help='Truncation psi for the trade-off between quality and diversity. Defaults to 1.', default=None)
parser.add_argument('--device', help='cpu|cuda', default='cuda')

args = parser.parse_args()

assert args.truncation is None

device = torch.device(args.device)

real = np.asarray(Image.open(args.image)).transpose([2, 0, 1])/255.0
masks = np.asarray(Image.open(args.mask).convert('1'), dtype=np.float32)

images = torch.Tensor(real.copy())[None,...]*2-1
masks = torch.Tensor(masks)[None,None,...].float()
masks = (masks>0).float()
latents_in = torch.randn(1, 512)

net = Generator()
net.load_state_dict(torch.load(args.checkpoint))
net.eval()

net = net.to(device)
images = images.to(device)
masks = masks.to(device)
latents_in = latents_in.to(device)

result = net(images, masks, [latents_in], truncation=args.truncation)
result = result.detach().cpu().numpy()
result = (result+1)/2
result = (result[0].transpose((1,2,0)))*255
Image.fromarray(result.clip(0,255).astype(np.uint8)).save(args.output)
