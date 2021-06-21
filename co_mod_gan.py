import pdb
from collections import OrderedDict
import torch
from torch import nn
from torch.nn import functional as F
from op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d
from stylegan2 import PixelNorm, EqualLinear, EqualConv2d, ConvLayer, StyledConv,ToRGB
import numpy as np

#----------------------------------------------------------------------------
# Mapping network.
# Transforms the input latent code (z) to the disentangled latent code (w).
# Used in configs B-F (Table 1).

class G_mapping(nn.Module):
    def __init__(self,
        latent_size             = 512,          # Latent vector (Z) dimensionality.
        label_size              = 0,            # Label dimensionality, 0 if no labels.
        dlatent_size            = 512,          # Disentangled latent (W) dimensionality.
        dlatent_broadcast       = None,         # Output disentangled latent (W) as [minibatch, dlatent_size] or [minibatch, dlatent_broadcast, dlatent_size].
        mapping_layers          = 8,            # Number of mapping layers.
        mapping_fmaps           = 512,          # Number of activations in the mapping layers.
        mapping_lrmul           = 0.01,         # Learning rate multiplier for the mapping layers.
        mapping_nonlinearity    = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
        normalize_latents       = True,         # Normalize latent vectors (Z) before feeding them to the mapping layers?
        **kwargs
        ):
        assert mapping_nonlinearity == 'lrelu'
        assert dlatent_broadcast is None
        super().__init__()
        layers = []

        # Embed labels and concatenate them with latents.
        if label_size:
            raise NotImplementedError

        # Normalize latents.
        if normalize_latents:
            layers.append(
                    ('Normalize', PixelNorm()))
        # Mapping layers.
        dim_in = latent_size
        for layer_idx in range(mapping_layers):
            fmaps = dlatent_size if layer_idx == mapping_layers - 1 else mapping_fmaps
            layers.append(
                    (
                        'Dense%d' % layer_idx,
                        EqualLinear(
                            dim_in,
                            fmaps,
                            lr_mul=mapping_lrmul,
                            activation="fused_lrelu")
                        ))
            dim_in = fmaps
        # Broadcast.
        if dlatent_broadcast is not None:
            raise NotImplementedError
        self.G_mapping = nn.Sequential(OrderedDict(layers))

    def forward(
            self,
            latents_in):
        styles = self.G_mapping(latents_in)
        return styles

#----------------------------------------------------------------------------
# CoModGAN synthesis network.

class G_synthesis_co_mod_gan(nn.Module):
    def __init__(
            self,
            dlatent_size        = 512,          # Disentangled latent (W) dimensionality.
            num_channels        = 3,            # Number of output color channels.
            resolution          = 512,         # Output resolution.
            fmap_base           = 16 << 10,     # Overall multiplier for the number of feature maps.
            fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
            fmap_min            = 1,            # Minimum number of feature maps in any layer.
            fmap_max            = 512,          # Maximum number of feature maps in any layer.
            randomize_noise     = True,         # True = randomize noise inputs every time (non-deterministic), False = read noise inputs from variables.
            architecture        = 'skip',       # Architecture: 'orig', 'skip', 'resnet'.
            nonlinearity        = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
            resample_kernel     = [1,3,3,1],    # Low-pass filter to apply when resampling activations. None = no filtering.
            fused_modconv       = True,         # Implement modulated_conv2d_layer() as a single fused op?
            pix2pix             = False,
            dropout_rate        = 0.5,
            cond_mod            = True,
            style_mod           = True,
            noise_injection     = True,
            **kwargs
            ):
        resolution_log2 = int(np.log2(resolution))
        assert resolution == 2**resolution_log2 and resolution >= 4
        def nf(stage): return np.clip(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_min, fmap_max)
        assert architecture in ['skip']
        assert nonlinearity == 'lrelu'
        assert fused_modconv
        assert cond_mod
        assert style_mod
        assert not pix2pix
        assert noise_injection

        super().__init__() 
        act = nonlinearity
        self.num_layers = resolution_log2 * 2 - 2
        self.resolution_log2 = resolution_log2

        class E_fromrgb(nn.Module): # res = 2..resolution_log2
            def __init__(self, res):
                super().__init__()
                self.FromRGB = ConvLayer(
                        num_channels+1,
                        nf(res-1),
                        1,
                        blur_kernel=resample_kernel,
                        activate=True)
            def forward(self, data):
                y, E_features = data
                t = self.FromRGB(y)
                return t, E_features
        class E_block(nn.Module): # res = 2..resolution_log2
            def __init__(self, res):
                super().__init__()
                self.Conv0 = ConvLayer(
                        nf(res-1),
                        nf(res-1),
                        kernel_size=3,
                        activate=True)
                self.Conv1_down = ConvLayer(
                        nf(res-1),
                        nf(res-2),
                        kernel_size=3,
                        downsample=True,
                        blur_kernel=resample_kernel,
                        activate=True)
                self.res = res
            def forward(self, data):
                x, E_features = data
                x = self.Conv0(x)
                E_features[self.res] = x
                x = self.Conv1_down(x)
                return x, E_features
        class E_block_final(nn.Module): # res = 2..resolution_log2
            def __init__(self):
                super().__init__()
                self.Conv = ConvLayer(
                        nf(2),
                        nf(1),
                        kernel_size=3,
                        activate=True)
                self.Dense0 = EqualLinear(nf(1)*4*4, nf(1)*2,
                                activation="fused_lrelu")
                self.dropout = nn.Dropout(dropout_rate)
            def forward(self, data):
                x, E_features = data
                x = self.Conv(x)
                E_features[2] = x
                bsize = x.size(0)
                x = x.view(bsize, -1)
                x = self.Dense0(x)
                x = self.dropout(x)
                return x, E_features

        # Main layers.
        Es = []
        for res in range(resolution_log2, 2, -1):
            if res == resolution_log2:
                Es.append(
                        (
                            '%dx%d_0' % (2**res, 2**res),
                            E_fromrgb(res)
                            ))
            Es.append(
                    (
                        '%dx%d' % (2**res, 2**res),
                        E_block(res)

                        ))
        # Final layers.
        Es.append(
                (
                    '4x4',
                    E_block_final()

                    ))
        self.E = nn.Sequential(OrderedDict(Es))

        # Single convolution layer with all the bells and whistles.
        # Building blocks for main layers.
        mod_size = 0
        if style_mod:
            mod_size += dlatent_size
        if cond_mod:
            mod_size += nf(1)*2
        assert mod_size > 0
        def get_mod(latent, x_global):
            mod_vector = []
            if style_mod:
                mod_vector.append(latent)
            if cond_mod:
                mod_vector.append(x_global)
            mod_vector = torch.cat(mod_vector, 1)
            return mod_vector
        class Block(nn.Module):
            def __init__(self, res):
                super().__init__()
                self.res = res
                self.Conv0_up = StyledConv(
                        nf(res-2),
                        nf(res-1),
                        kernel_size=3,
                        style_dim=mod_size,
                        upsample=True,
                        blur_kernel=resample_kernel)
                self.Conv1 = StyledConv(
                        nf(res-1),
                        nf(res-1),
                        kernel_size=3,
                        style_dim=mod_size,
                        upsample=False)
                self.ToRGB = ToRGB(
                        nf(res-1),
                        mod_size)
            def forward(self, x, y, dlatents_in, x_global, E_features):
                x_skip = E_features[self.res]
                mod_vector = get_mod(dlatents_in[:,self.res*2-5], x_global)
                x = self.Conv0_up(x, mod_vector)
                x = x + x_skip
                mod_vector = get_mod(dlatents_in[:,self.res*2-4], x_global)
                x = self.Conv1(x, mod_vector)
                mod_vector = get_mod(dlatents_in[:,self.res*2-3], x_global)
                y = self.ToRGB(x, mod_vector, skip=y)
                return x, y
        class Block0(nn.Module):
            def __init__(self):
                super().__init__()
                self.Dense = EqualLinear(
                        nf(1)*2,
                        nf(1)*4*4,
                        activation="fused_lrelu")
                self.Conv = StyledConv(
                        nf(1),
                        nf(1),
                        kernel_size=3,
                        style_dim=mod_size)
                self.ToRGB = ToRGB(
                        nf(1),
                        style_dim=mod_size,
                        upsample=False)
            def forward(self, x, dlatents_in, x_global):
                x = self.Dense(x)
                x = x.view(-1, nf(1), 4, 4)
                mod_vector = get_mod(dlatents_in[:,0], x_global)
                x = self.Conv(x, mod_vector)
                mod_vector = get_mod(dlatents_in[:,1], x_global)
                y = self.ToRGB(x, mod_vector)
                return x, y
        # Early layers.
        self.G_4x4 = Block0()
        # Main layers.
        for res in range(3, resolution_log2 + 1):
            setattr(self, 'G_%dx%d' % (2**res, 2**res),
                    Block(res))

    def forward(self, images_in, masks_in, dlatents_in):
        y = torch.cat([masks_in - 0.5, images_in * masks_in], 1)
        E_features = {}
        x_global, E_features = self.E((y, E_features))
        x = x_global
        x, y = self.G_4x4(x, dlatents_in, x_global)
        for res in range(3, self.resolution_log2 + 1):
            block = getattr(self, 'G_%dx%d' % (2**res, 2**res))
            x, y = block(x, y, dlatents_in, x_global, E_features)
        images_out = y * (1 - masks_in) + images_in * masks_in
        return images_out

#----------------------------------------------------------------------------
# Main generator network.
# Composed of two sub-networks (mapping and synthesis) that are defined below.
# Used in configs B-F (Table 1).

class Generator(nn.Module):
    def __init__(
            self,
            **kwargs):                                          # Arguments for sub-networks (mapping and synthesis).
        super().__init__()
        self.G_mapping = G_mapping(**kwargs)
        self.G_synthesis = G_synthesis_co_mod_gan(**kwargs)

    def forward(
            self,
            images_in,
            masks_in,
            latents_in,
            return_latents=False,
            inject_index=None,
            truncation=0.5,
            truncation_mean=None,
            ):
        assert isinstance(latents_in, list)
        dlatents_in = [self.G_mapping(s) for s in latents_in]
        if truncation is not None:
            dlatents_t = []
            for style in dlatents_in:
                dlatents_t.append(
                    truncation_mean + truncation * (style - truncation_mean)
                )
            dlatents_in = dlatents_t
        if len(dlatents_in) < 2:
            inject_index = self.G_synthesis.num_layers
            if dlatents_in[0].ndim < 3:
                dlatent = dlatents_in[0].unsqueeze(1).repeat(1, inject_index, 1)
            else:
                dlatent = dlatents_in[0]
        else:
            if inject_index is None:
                inject_index = random.randint(1, self.G_synthesis.num_layers - 1)
            dlatent = dlatents_in[0].unsqueeze(1).repeat(1, inject_index, 1)
            dlatent2 = dlatents_in[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)

            dlatent = torch.cat([dlatent, dlatent2], 1)
        output = self.G_synthesis(images_in, masks_in, dlatent)
        return output
