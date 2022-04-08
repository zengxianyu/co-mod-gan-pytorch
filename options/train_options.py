"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.add_argument('--save_remote_gs', type=str, required=False)
        parser.add_argument('--trainer', type=str, default='stylegan2')
        # for displays
        parser.add_argument('--display_freq', type=int, default=101, help='frequency of showing training results on screen')
        parser.add_argument('--print_freq', type=int, default=101, help='frequency of showing training results on console')
        parser.add_argument('--save_latest_freq', type=int, default=50000, help='frequency of saving the latest results')
        parser.add_argument('--validation_freq', type=int, default=50000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=10, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--debug', action='store_true', help='only do one epoch and displays at each iteration')
        #  datast
        parser.add_argument('--dataset_mode_train', type=str, default='coco')
        parser.add_argument('--dataset_mode', type=str, default='coco')
        parser.add_argument('--dataset_mode_val', type=str, required=False)

        # for training
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--niter', type=int, default=50, help='# of iter at starting learning rate. This is NOT the total #epochs. Totla #epochs is niter + niter_decay')
        parser.add_argument('--niter_decay', type=int, default=0, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--optimizer', type=str, default='adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--D_steps_per_G', type=int, default=1, help='number of discriminator iterations per generator iterations.')

        # for discriminators
        parser.add_argument('--lambda_vgg', type=float, default=10.0, help='weight for vgg loss')
        parser.add_argument('--lambda_l1', type=float, default=1.0, help='weight for l1 loss')
        parser.add_argument('--no_l1_loss', action='store_true', help='if specified, do *not* use l1 loss')
        parser.add_argument('--no_vgg_loss', action='store_true', help='if specified, do *not* use VGG feature matching loss')
        parser.add_argument('--gan_mode', type=str, default='hinge', help='(ls|original|hinge)')
        parser.add_argument('--netD', type=str, default='comodgan')
        parser.add_argument('--freeze_D', action='store_true', help='do not update D')
        self.isTrain = True
        return parser
