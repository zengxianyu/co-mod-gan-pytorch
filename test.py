"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import pdb
import cv2
import os
from collections import OrderedDict
import json
from tqdm import tqdm
import numpy as np
import torch
import data
from options.test_options import TestOptions
#from models.pix2pix_model import Pix2PixModel
import models


opt = TestOptions().parse()

dataloader = data.create_dataloader(opt)

model = models.create_model(opt)
model.eval()

for i, data_i in tqdm(enumerate(dataloader)):
    if i * opt.batchSize >= opt.how_many:
        break
    with torch.no_grad():
        generated,_ = model(data_i, mode='inference')
    generated = torch.clamp(generated, -1, 1)
    generated = (generated+1)/2*255
    generated = generated.cpu().numpy().astype(np.uint8)
    img_path = data_i['path']
    for b in range(generated.shape[0]):
        pred_im = generated[b].transpose((1,2,0))
        print('process image... %s' % img_path[b])
        cv2.imwrite(img_path[b], pred_im[:,:,::-1])
