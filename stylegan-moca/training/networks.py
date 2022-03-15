# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import os.path

import numpy as np
import random
import torch
import torch.distributed as dist
from sklearn.cluster import KMeans
import pickle

from torch_utils import misc
from torch_utils import persistence
from torch_utils.ops import conv2d_resample
from torch_utils.ops import upfirdn2d
from torch_utils.ops import bias_act
from torch_utils.ops import fma


#----------------------------------------------------------------------------

@misc.profiled_function
def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()

#----------------------------------------------------------------------------

@misc.profiled_function
def modulated_conv2d(
    x,                          # Input tensor of shape [batch_size, in_channels, in_height, in_width].
    weight,                     # Weight tensor of shape [out_channels, in_channels, kernel_height, kernel_width].
    styles,                     # Modulation coefficients of shape [batch_size, in_channels].
    noise           = None,     # Optional noise tensor to add to the output activations.
    up              = 1,        # Integer upsampling factor.
    down            = 1,        # Integer downsampling factor.
    padding         = 0,        # Padding with respect to the upsampled image.
    resample_filter = None,     # Low-pass filter to apply when resampling activations. Must be prepared beforehand by calling upfirdn2d.setup_filter().
    demodulate      = True,     # Apply weight demodulation?
    flip_weight     = True,     # False = convolution, True = correlation (matches torch.nn.functional.conv2d).
    fused_modconv   = True,     # Perform modulation, convolution, and demodulation as a single fused operation?
):
    batch_size = x.shape[0]
    out_channels, in_channels, kh, kw = weight.shape
    misc.assert_shape(weight, [out_channels, in_channels, kh, kw]) # [OIkk]
    misc.assert_shape(x, [batch_size, in_channels, None, None]) # [NIHW]
    misc.assert_shape(styles, [batch_size, in_channels]) # [NI]

    # Pre-normalize inputs to avoid FP16 overflow.
    if x.dtype == torch.float16 and demodulate:
        weight = weight * (1 / np.sqrt(in_channels * kh * kw) / weight.norm(float('inf'), dim=[1,2,3], keepdim=True)) # max_Ikk
        styles = styles / styles.norm(float('inf'), dim=1, keepdim=True) # max_I

    # Calculate per-sample weights and demodulation coefficients.
    w = None
    dcoefs = None
    if demodulate or fused_modconv:
        w = weight.unsqueeze(0) # [NOIkk]
        w = w * styles.reshape(batch_size, 1, -1, 1, 1) # [NOIkk]
    if demodulate:
        dcoefs = (w.square().sum(dim=[2,3,4]) + 1e-8).rsqrt() # [NO]
    if demodulate and fused_modconv:
        w = w * dcoefs.reshape(batch_size, -1, 1, 1, 1) # [NOIkk]

    # Execute by scaling the activations before and after the convolution.
    if not fused_modconv:
        x = x * styles.to(x.dtype).reshape(batch_size, -1, 1, 1)
        x = conv2d_resample.conv2d_resample(x=x, w=weight.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, flip_weight=flip_weight)
        if demodulate and noise is not None:
            x = fma.fma(x, dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1), noise.to(x.dtype))
        elif demodulate:
            x = x * dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1)
        elif noise is not None:
            x = x.add_(noise.to(x.dtype))
        return x

    # Execute as one fused op using grouped convolution.
    with misc.suppress_tracer_warnings(): # this value will be treated as a constant
        batch_size = int(batch_size)
    misc.assert_shape(x, [batch_size, in_channels, None, None])
    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, in_channels, kh, kw)
    x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, groups=batch_size, flip_weight=flip_weight)
    x = x.reshape(batch_size, -1, *x.shape[2:])
    if noise is not None:
        x = x.add_(noise)
    return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class FullyConnectedLayer(torch.nn.Module):
    def __init__(self,
        in_features,                # Number of input features.
        out_features,               # Number of output features.
        bias            = True,     # Apply additive bias before the activation function?
        activation      = 'linear', # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 1,        # Learning rate multiplier.
        bias_init       = 0,        # Initial value for the additive bias.
    ):
        super().__init__()
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = torch.nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class Conv2dLayer(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        kernel_size,                    # Width and height of the convolution kernel.
        bias            = True,         # Apply additive bias before the activation function?
        activation      = 'linear',     # Activation function: 'relu', 'lrelu', etc.
        up              = 1,            # Integer upsampling factor.
        down            = 1,            # Integer downsampling factor.
        resample_filter = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp      = None,         # Clamp the output to +-X, None = disable clamping.
        channels_last   = False,        # Expect the input to have memory_format=channels_last?
        trainable       = True,         # Update the weights of this layer during training?
    ):
        super().__init__()
        self.activation = activation
        self.up = up
        self.down = down
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))
        self.act_gain = bias_act.activation_funcs[activation].def_gain

        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        weight = torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format)
        bias = torch.zeros([out_channels]) if bias else None
        if trainable:
            self.weight = torch.nn.Parameter(weight)
            self.bias = torch.nn.Parameter(bias) if bias is not None else None
        else:
            self.register_buffer('weight', weight)
            if bias is not None:
                self.register_buffer('bias', bias)
            else:
                self.bias = None

    def forward(self, x, gain=1):
        w = self.weight * self.weight_gain
        b = self.bias.to(x.dtype) if self.bias is not None else None
        flip_weight = (self.up == 1) # slightly faster
        x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=self.resample_filter, up=self.up, down=self.down, padding=self.padding, flip_weight=flip_weight)

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act.bias_act(x, b, act=self.activation, gain=act_gain, clamp=act_clamp)
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class MappingNetwork(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality, 0 = no latent.
        c_dim,                      # Conditioning label (C) dimensionality, 0 = no label.
        w_dim,                      # Intermediate latent (W) dimensionality.
        num_ws,                     # Number of intermediate latents to output, None = do not broadcast.
        num_layers      = 8,        # Number of mapping layers.
        embed_features  = None,     # Label embedding dimensionality, None = same as w_dim.
        layer_features  = None,     # Number of intermediate features in the mapping layers, None = same as w_dim.
        activation      = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 0.01,     # Learning rate multiplier for the mapping layers.
        w_avg_beta      = 0.995,    # Decay for tracking the moving average of W during training, None = do not track.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        if embed_features is None:
            embed_features = w_dim
        if c_dim == 0:
            embed_features = 0
        if layer_features is None:
            layer_features = w_dim
        features_list = [z_dim + embed_features] + [layer_features] * (num_layers - 1) + [w_dim]

        if c_dim > 0:
            self.embed = FullyConnectedLayer(c_dim, embed_features)
        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = FullyConnectedLayer(in_features, out_features, activation=activation, lr_multiplier=lr_multiplier)
            setattr(self, f'fc{idx}', layer)

        if num_ws is not None and w_avg_beta is not None:
            self.register_buffer('w_avg', torch.zeros([w_dim]))

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, skip_w_avg_update=False):
        # Embed, normalize, and concat inputs.
        x = None
        with torch.autograd.profiler.record_function('input'):
            if self.z_dim > 0:
                misc.assert_shape(z, [None, self.z_dim])
                x = normalize_2nd_moment(z.to(torch.float32))
            if self.c_dim > 0:
                misc.assert_shape(c, [None, self.c_dim])
                y = normalize_2nd_moment(self.embed(c.to(torch.float32)))
                x = torch.cat([x, y], dim=1) if x is not None else y

        # Main layers.
        for idx in range(self.num_layers):
            layer = getattr(self, f'fc{idx}')
            x = layer(x)

        # Update moving average of W.
        if self.w_avg_beta is not None and self.training and not skip_w_avg_update:
            with torch.autograd.profiler.record_function('update_w_avg'):
                self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

        # Broadcast.
        if self.num_ws is not None:
            with torch.autograd.profiler.record_function('broadcast'):
                x = x.unsqueeze(1).repeat([1, self.num_ws, 1])

        # Apply truncation.
        if truncation_psi != 1:
            with torch.autograd.profiler.record_function('truncate'):
                assert self.w_avg_beta is not None
                if self.num_ws is None or truncation_cutoff is None:
                    x = self.w_avg.lerp(x, truncation_psi)
                else:
                    x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisLayer(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        w_dim,                          # Intermediate latent (W) dimensionality.
        resolution,                     # Resolution of this layer.
        kernel_size     = 3,            # Convolution kernel size.
        up              = 1,            # Integer upsampling factor.
        use_noise       = True,         # Enable noise input?
        activation      = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
        resample_filter = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp      = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        channels_last   = False,        # Use channels_last format for the weights?
    ):
        super().__init__()
        self.resolution = resolution
        self.up = up
        self.use_noise = use_noise
        self.activation = activation
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.act_gain = bias_act.activation_funcs[activation].def_gain

        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format))
        if use_noise:
            self.register_buffer('noise_const', torch.randn([resolution, resolution]))
            self.noise_strength = torch.nn.Parameter(torch.zeros([]))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))

    def forward(self, x, w, noise_mode='random', fused_modconv=True, gain=1):
        assert noise_mode in ['random', 'const', 'none']
        in_resolution = self.resolution // self.up
        misc.assert_shape(x, [None, self.weight.shape[1], in_resolution, in_resolution])
        styles = self.affine(w)

        noise = None
        if self.use_noise and noise_mode == 'random':
            noise = torch.randn([x.shape[0], 1, self.resolution, self.resolution], device=x.device) * self.noise_strength
        if self.use_noise and noise_mode == 'const':
            noise = self.noise_const * self.noise_strength

        flip_weight = (self.up == 1) # slightly faster
        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, noise=noise, up=self.up,
            padding=self.padding, resample_filter=self.resample_filter, flip_weight=flip_weight, fused_modconv=fused_modconv)

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act.bias_act(x, self.bias.to(x.dtype), act=self.activation, gain=act_gain, clamp=act_clamp)
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class ToRGBLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, w_dim, kernel_size=1, conv_clamp=None, channels_last=False):
        super().__init__()
        self.conv_clamp = conv_clamp
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))

    def forward(self, x, w, fused_modconv=True):
        styles = self.affine(w) * self.weight_gain
        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, demodulate=False, fused_modconv=fused_modconv)
        x = bias_act.bias_act(x, self.bias.to(x.dtype), clamp=self.conv_clamp)
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisBlock(torch.nn.Module):
    def __init__(self,
        in_channels,                        # Number of input channels, 0 = first block.
        out_channels,                       # Number of output channels.
        w_dim,                              # Intermediate latent (W) dimensionality.
        resolution,                         # Resolution of this block.
        img_channels,                       # Number of output color channels.
        is_last,                            # Is this the last block?
        architecture        = 'skip',       # Architecture: 'orig', 'skip', 'resnet'.
        resample_filter     = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp          = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16            = False,        # Use FP16 for this block?
        fp16_channels_last  = False,        # Use channels-last memory format with FP16?
        **layer_kwargs,                     # Arguments for SynthesisLayer.
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.is_last = is_last
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.num_conv = 0
        self.num_torgb = 0

        if in_channels == 0:
            self.const = torch.nn.Parameter(torch.randn([out_channels, resolution, resolution]))

        if in_channels != 0:
            self.conv0 = SynthesisLayer(in_channels, out_channels, w_dim=w_dim, resolution=resolution, up=2,
                resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs)
            self.num_conv += 1

        self.conv1 = SynthesisLayer(out_channels, out_channels, w_dim=w_dim, resolution=resolution,
            conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs)
        self.num_conv += 1

        if is_last or architecture == 'skip':
            self.torgb = ToRGBLayer(out_channels, img_channels, w_dim=w_dim,
                conv_clamp=conv_clamp, channels_last=self.channels_last)
            self.num_torgb += 1

        if in_channels != 0 and architecture == 'resnet':
            self.skip = Conv2dLayer(in_channels, out_channels, kernel_size=1, bias=False, up=2,
                resample_filter=resample_filter, channels_last=self.channels_last)

    def forward(self, x, img, ws, force_fp32=False, fused_modconv=None, **layer_kwargs):
        misc.assert_shape(ws, [None, self.num_conv + self.num_torgb, self.w_dim])
        w_iter = iter(ws.unbind(dim=1))
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format
        if fused_modconv is None:
            with misc.suppress_tracer_warnings(): # this value will be treated as a constant
                fused_modconv = (not self.training) and (dtype == torch.float32 or int(x.shape[0]) == 1)

        # Input.
        if self.in_channels == 0:
            x = self.const.to(dtype=dtype, memory_format=memory_format)
            x = x.unsqueeze(0).repeat([ws.shape[0], 1, 1, 1])
        else:
            misc.assert_shape(x, [None, self.in_channels, self.resolution // 2, self.resolution // 2])
            x = x.to(dtype=dtype, memory_format=memory_format)

        # Main layers.
        if self.in_channels == 0:
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
        elif self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, gain=np.sqrt(0.5), **layer_kwargs)
            x = y.add_(x)
        else:
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)

        # ToRGB.
        if img is not None:
            misc.assert_shape(img, [None, self.img_channels, self.resolution // 2, self.resolution // 2])
            img = upfirdn2d.upsample2d(img, self.resample_filter)
        if self.is_last or self.architecture == 'skip':
            y = self.torgb(x, next(w_iter), fused_modconv=fused_modconv)
            y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            img = img.add_(y) if img is not None else y

        assert x.dtype == dtype
        assert img is None or img.dtype == torch.float32
        return x, img

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisNetwork(torch.nn.Module):
    def __init__(self,
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output image resolution.
        img_channels,               # Number of color channels.
        channel_base    = 32768,    # Overall multiplier for the number of channels.
        channel_max     = 512,      # Maximum number of channels in any layer.
        num_fp16_res    = 0,        # Use FP16 for the N highest resolutions.
        **block_kwargs,             # Arguments for SynthesisBlock.
    ):
        assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0
        super().__init__()
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(2, self.img_resolution_log2 + 1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        self.num_ws = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res // 2] if res > 4 else 0
            out_channels = channels_dict[res]
            use_fp16 = (res >= fp16_resolution)
            is_last = (res == self.img_resolution)
            block = SynthesisBlock(in_channels, out_channels, w_dim=w_dim, resolution=res,
                img_channels=img_channels, is_last=is_last, use_fp16=use_fp16, **block_kwargs)
            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb
            setattr(self, f'b{res}', block)

    def forward(self, ws, **block_kwargs):
        block_ws = []
        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
            ws = ws.to(torch.float32)
            w_idx = 0
            for res in self.block_resolutions:
                block = getattr(self, f'b{res}')
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv

        x = img = None
        for res, cur_ws in zip(self.block_resolutions, block_ws):
            block = getattr(self, f'b{res}')
            x, img = block(x, img, cur_ws, **block_kwargs)
        return img

#----------------------------------------------------------------------------

@persistence.persistent_class
class Generator(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        synthesis_kwargs    = {},   # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.synthesis = SynthesisNetwork(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels, **synthesis_kwargs)
        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, **synthesis_kwargs):
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        img = self.synthesis(ws, **synthesis_kwargs)
        return img

#----------------------------------------------------------------------------

@persistence.persistent_class
class DiscriminatorBlock(torch.nn.Module):
    def __init__(self,
        in_channels,                        # Number of input channels, 0 = first block.
        tmp_channels,                       # Number of intermediate channels.
        out_channels,                       # Number of output channels.
        resolution,                         # Resolution of this block.
        img_channels,                       # Number of input color channels.
        first_layer_idx,                    # Index of the first layer.
        architecture        = 'resnet',     # Architecture: 'orig', 'skip', 'resnet'.
        activation          = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
        resample_filter     = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp          = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16            = False,        # Use FP16 for this block?
        fp16_channels_last  = False,        # Use channels-last memory format with FP16?
        freeze_layers       = 0,            # Freeze-D: Number of layers to freeze.
    ):
        assert in_channels in [0, tmp_channels]
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.resolution = resolution
        self.img_channels = img_channels
        self.first_layer_idx = first_layer_idx
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))

        self.num_layers = 0
        def trainable_gen():
            while True:
                layer_idx = self.first_layer_idx + self.num_layers
                trainable = (layer_idx >= freeze_layers)
                self.num_layers += 1
                yield trainable
        trainable_iter = trainable_gen()

        if in_channels == 0 or architecture == 'skip':
            self.fromrgb = Conv2dLayer(img_channels, tmp_channels, kernel_size=1, activation=activation,
                trainable=next(trainable_iter), conv_clamp=conv_clamp, channels_last=self.channels_last)

        self.conv0 = Conv2dLayer(tmp_channels, tmp_channels, kernel_size=3, activation=activation,
            trainable=next(trainable_iter), conv_clamp=conv_clamp, channels_last=self.channels_last)

        self.conv1 = Conv2dLayer(tmp_channels, out_channels, kernel_size=3, activation=activation, down=2,
            trainable=next(trainable_iter), resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=self.channels_last)

        if architecture == 'resnet':
            self.skip = Conv2dLayer(tmp_channels, out_channels, kernel_size=1, bias=False, down=2,
                trainable=next(trainable_iter), resample_filter=resample_filter, channels_last=self.channels_last)

    def forward(self, x, img, force_fp32=False):
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format

        # Input.
        if x is not None:
            misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution])
            x = x.to(dtype=dtype, memory_format=memory_format)

        # FromRGB.
        if self.in_channels == 0 or self.architecture == 'skip':
            misc.assert_shape(img, [None, self.img_channels, self.resolution, self.resolution])
            img = img.to(dtype=dtype, memory_format=memory_format)
            y = self.fromrgb(img)
            x = x + y if x is not None else y
            img = upfirdn2d.downsample2d(img, self.resample_filter) if self.architecture == 'skip' else None

        # Main layers.
        if self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x)
            x = self.conv1(x, gain=np.sqrt(0.5))
            x = y.add_(x)
        else:
            x = self.conv0(x)
            x = self.conv1(x)

        assert x.dtype == dtype
        return x, img

#----------------------------------------------------------------------------

@persistence.persistent_class
class MinibatchStdLayer(torch.nn.Module):
    def __init__(self, group_size, num_channels=1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x):
        N, C, H, W = x.shape
        with misc.suppress_tracer_warnings(): # as_tensor results are registered as constants
            G = torch.min(torch.as_tensor(self.group_size), torch.as_tensor(N)) if self.group_size is not None else N
        F = self.num_channels
        c = C // F

        y = x.reshape(G, -1, F, c, H, W)    # [GnFcHW] Split minibatch N into n groups of size G, and channels C into F groups of size c.
        y = y - y.mean(dim=0)               # [GnFcHW] Subtract mean over group.
        y = y.square().mean(dim=0)          # [nFcHW]  Calc variance over group.
        y = (y + 1e-8).sqrt()               # [nFcHW]  Calc stddev over group.
        y = y.mean(dim=[2,3,4])             # [nF]     Take average over channels and pixels.
        y = y.reshape(-1, F, 1, 1)          # [nF11]   Add missing dimensions.
        y = y.repeat(G, 1, H, W)            # [NFHW]   Replicate over group and pixels.
        x = torch.cat([x, y], dim=1)        # [NCHW]   Append to input as new channels.
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class DiscriminatorEpilogue(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        cmap_dim,                       # Dimensionality of mapped conditioning label, 0 = no label.
        resolution,                     # Resolution of this block.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        mbstd_group_size    = 4,        # Group size for the minibatch standard deviation layer, None = entire minibatch.
        mbstd_num_channels  = 1,        # Number of features for the minibatch standard deviation layer, 0 = disable.
        activation          = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        conv_clamp          = None,     # Clamp the output of convolution layers to +-X, None = disable clamping.
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.cmap_dim = cmap_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.architecture = architecture

        if architecture == 'skip':
            self.fromrgb = Conv2dLayer(img_channels, in_channels, kernel_size=1, activation=activation)
        self.mbstd = MinibatchStdLayer(group_size=mbstd_group_size, num_channels=mbstd_num_channels) if mbstd_num_channels > 0 else None
        self.conv = Conv2dLayer(in_channels + mbstd_num_channels, in_channels, kernel_size=3, activation=activation, conv_clamp=conv_clamp)
        self.fc = FullyConnectedLayer(in_channels * (resolution ** 2), in_channels, activation=activation)
        self.out = FullyConnectedLayer(in_channels, 1 if cmap_dim == 0 else cmap_dim)

    def forward(self, x, img, cmap, force_fp32=False):
        misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution]) # [NCHW]
        _ = force_fp32 # unused
        dtype = torch.float32
        memory_format = torch.contiguous_format

        # FromRGB.
        x = x.to(dtype=dtype, memory_format=memory_format)
        if self.architecture == 'skip':
            misc.assert_shape(img, [None, self.img_channels, self.resolution, self.resolution])
            img = img.to(dtype=dtype, memory_format=memory_format)
            x = x + self.fromrgb(img)

        # Main layers.
        if self.mbstd is not None:
            x = self.mbstd(x)
        x = self.conv(x)
        x = self.fc(x.flatten(1))
        x = self.out(x)

        # Conditioning.
        if self.cmap_dim > 0:
            misc.assert_shape(cmap, [None, self.cmap_dim])
            x = (x * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))

        assert x.dtype == dtype
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class Discriminator(torch.nn.Module):
    def __init__(self,
        c_dim,                          # Conditioning label (C) dimensionality.
        img_resolution,                 # Input resolution.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        channel_base        = 32768,    # Overall multiplier for the number of channels.
        channel_max         = 512,      # Maximum number of channels in any layer.
        num_fp16_res        = 0,        # Use FP16 for the N highest resolutions.
        conv_clamp          = None,     # Clamp the output of convolution layers to +-X, None = disable clamping.
        cmap_dim            = None,     # Dimensionality of mapped conditioning label, None = default.
        block_kwargs        = {},       # Arguments for DiscriminatorBlock.
        mapping_kwargs      = {},       # Arguments for MappingNetwork.
        epilogue_kwargs     = {},       # Arguments for DiscriminatorEpilogue.
    ):
        super().__init__()
        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        if cmap_dim is None:
            cmap_dim = channels_dict[4]
        if c_dim == 0:
            cmap_dim = 0

        common_kwargs = dict(img_channels=img_channels, architecture=architecture, conv_clamp=conv_clamp)
        cur_layer_idx = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res] if res < img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            use_fp16 = (res >= fp16_resolution)
            block = DiscriminatorBlock(in_channels, tmp_channels, out_channels, resolution=res,
                first_layer_idx=cur_layer_idx, use_fp16=use_fp16, **block_kwargs, **common_kwargs)
            setattr(self, f'b{res}', block)
            cur_layer_idx += block.num_layers
        if c_dim > 0:
            self.mapping = MappingNetwork(z_dim=0, c_dim=c_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None, **mapping_kwargs)
        self.b4 = DiscriminatorEpilogue(channels_dict[4], cmap_dim=cmap_dim, resolution=4, **epilogue_kwargs, **common_kwargs)

    def forward(self, img, c, **block_kwargs):
        x = None
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            x, img = block(x, img, **block_kwargs)

        cmap = None
        if self.c_dim > 0:
            cmap = self.mapping(None, c)
        x = self.b4(x, img, cmap)
        return x

#----------------------------------------------------------------------------
#################################################
#        VAE ENCODER based on Discriminator     #
#################################################

# some hacky module
@persistence.persistent_class
class EncoderEpilogue(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        cmap_dim,                       # Dimensionality of mapped conditioning label, 0 = no label.
        resolution,                     # Resolution of this block.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        mbstd_group_size    = 4,        # Group size for the minibatch standard deviation layer, None = entire minibatch.
        mbstd_num_channels  = 1,        # Number of features for the minibatch standard deviation layer, 0 = disable.
        activation          = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        conv_clamp          = None,     # Clamp the output of convolution layers to +-X, None = disable clamping.
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cmap_dim = cmap_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.architecture = architecture

        if architecture == 'skip':
            self.fromrgb = Conv2dLayer(img_channels, in_channels, kernel_size=1, activation=activation)
        self.mbstd = MinibatchStdLayer(group_size=mbstd_group_size, num_channels=mbstd_num_channels) if mbstd_num_channels > 0 else None
        self.conv = Conv2dLayer(in_channels + mbstd_num_channels, in_channels, kernel_size=3, activation=activation, conv_clamp=conv_clamp)
        self.fc = FullyConnectedLayer(in_channels * (resolution ** 2), in_channels, activation=activation)
        self.out = FullyConnectedLayer(in_channels, out_channels)

    def forward(self, x, img, cmap, force_fp32=False):
        misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution]) # [NCHW]
        _ = force_fp32 # unused
        dtype = torch.float32
        memory_format = torch.contiguous_format

        # FromRGB.
        x = x.to(dtype=dtype, memory_format=memory_format)
        if self.architecture == 'skip':
            misc.assert_shape(img, [None, self.img_channels, self.resolution, self.resolution])
            img = img.to(dtype=dtype, memory_format=memory_format)
            x = x + self.fromrgb(img)

        # Main layers.
        if self.mbstd is not None:
            x = self.mbstd(x)
        x = self.conv(x)
        x = self.fc(x.flatten(1))
        x = self.out(x)

        # Conditioning.
        if self.cmap_dim > 0:
            misc.assert_shape(cmap, [None, self.cmap_dim])
            x = (x * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))

        assert x.dtype == dtype
        return x

@persistence.persistent_class
class Encoder(torch.nn.Module):
    """directly based on the architecture of discriminator"""
    def __init__(self,
        z_dim,                          # latent code dimensionality
        c_dim,                          # Conditioning label (C) dimensionality.
        img_resolution,                 # Input resolution.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        channel_base        = 32768,    # Overall multiplier for the number of channels.
        channel_max         = 512,      # Maximum number of channels in any layer.
        num_fp16_res        = 0,        # Use FP16 for the N highest resolutions.
        conv_clamp          = None,     # Clamp the output of convolution layers to +-X, None = disable clamping.
        cmap_dim            = None,     # Dimensionality of mapped conditioning label, None = default.
        block_kwargs        = {},       # Arguments for DiscriminatorBlock.
        mapping_kwargs      = {},       # Arguments for MappingNetwork.
        epilogue_kwargs     = {},       # Arguments for DiscriminatorEpilogue.
    ):
        super().__init__()
        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]

        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        if cmap_dim is None:
            cmap_dim = channels_dict[4]
        if c_dim == 0:
            cmap_dim = 0

        common_kwargs = dict(img_channels=img_channels, architecture=architecture, conv_clamp=conv_clamp)
        cur_layer_idx = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res] if res < img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            use_fp16 = (res >= fp16_resolution)
            block = DiscriminatorBlock(in_channels, tmp_channels, out_channels, resolution=res,
                first_layer_idx=cur_layer_idx, use_fp16=use_fp16, **block_kwargs, **common_kwargs)
            setattr(self, f'b{res}', block)
            cur_layer_idx += block.num_layers
        if c_dim > 0:
            self.mapping = MappingNetwork(z_dim=0, c_dim=c_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None,
                                          **mapping_kwargs)

        self.b4 = EncoderEpilogue(channels_dict[4], z_dim,
                                  cmap_dim=cmap_dim, resolution=4,
                                  **epilogue_kwargs, **common_kwargs)
        self.mean_linear = torch.nn.Linear(z_dim, z_dim)
        self.logvar_linear = torch.nn.Linear(z_dim, z_dim)
        self.logvar_linear.weight.data.fill_(0.)

    def forward(self, img, c, **block_kwargs):
        x = None
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            x, img = block(x, img, **block_kwargs)
        cmap = None
        if self.c_dim > 0:
            cmap = self.mapping(None, c)
        x = self.b4(x, img, cmap)
        self.mean = self.mean_linear(x)
        self.logvar = self.logvar_linear(x)

        # reparameterized trick
        # print("mean before repraram", self.mean)
        z_sample = self.reparameter(self.mean, self.logvar) # (N, latent_dim)
        return z_sample, self.mean, self.logvar

    def reparameter(self, mean, logvar):
        """reparameter and return a sampled feature veector"""
        # reparameterization trick for VAE
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps * std + mean
        return z


@persistence.persistent_class
class ResBlock(torch.nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = torch.nn.Sequential(
            Conv2dLayer(in_channel, channel, 3, activation='lrelu'),
            Conv2dLayer(channel, in_channel, 1, activation='lrelu'),
        )
        self.skip = Conv2dLayer(in_channel, in_channel, 1)

    def forward(self, input):
        res = self.skip(input)
        out = self.conv(input)
        out = out + res

        return out


@persistence.persistent_class
class SimpleEncoder(torch.nn.Module):
    def __init__(self,
                 z_dim,
                 c_dim,  # Conditioning label (C) dimensionality.
                 img_resolution,
                 img_channels,
                 channel=256,
                 n_res_block=8,
                 n_res_channel=128,

                 # these are all useless args, why they are here? just not to violate original signature

                 architecture='resnet',  # Architecture: 'orig', 'skip', 'resnet'.
                 channel_base=32768,  # Overall multiplier for the number of channels.
                 channel_max=512,  # Maximum number of channels in any layer.
                 num_fp16_res=0,  # Use FP16 for the N highest resolutions.
                 conv_clamp=None,  # Clamp the output of convolution layers to +-X, None = disable clamping.
                 cmap_dim=None,  # Dimensionality of mapped conditioning label, None = default.
                 block_kwargs={},  # Arguments for DiscriminatorBlock.
                 mapping_kwargs={},  # Arguments for MappingNetwork.
                 epilogue_kwargs={},  # Arguments for DiscriminatorEpilogue.
                 ):
        super().__init__()

        if img_resolution >= 512:
            stride = 4
            ds_layer_num = int(np.log2(img_resolution // 2)-2)

        else:
            stride = 2
            ds_layer_num = int(np.log2(img_resolution // 2)-1)

        if stride == 4:
            blocks = [
                Conv2dLayer(img_channels, channel // 2, 4, down=2, activation='lrelu'),
                Conv2dLayer(channel // 2, channel // 2, 4, down=2, activation='lrelu'),
                Conv2dLayer(channel // 2, channel, 3),
            ]

        elif stride == 2:
            blocks = [
                Conv2dLayer(img_channels, channel // 2, 4, down=2, activation='lrelu'),
                Conv2dLayer(channel // 2, channel, 3),
            ]
        else:
            raise Exception(f'Unsupported stride {stride}')

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        for i in range(ds_layer_num):
            block = [
                Conv2dLayer(channel, channel // 2, 4, down=2, activation='lrelu'),
                Conv2dLayer(channel // 2, channel, 3,  activation='lrelu'),
            ]
            blocks += block
        self.blocks = torch.nn.Sequential(*blocks)
        final_channel = channel
        self.fc = FullyConnectedLayer(final_channel*4, z_dim, activation='linear', lr_multiplier=0.01)
        self.mean_linear = FullyConnectedLayer(z_dim, z_dim, activation='linear', lr_multiplier=0.01)
        self.logvar_linear = FullyConnectedLayer(z_dim, z_dim, activation='linear', lr_multiplier=0.01)
        self.logvar_linear.weight.data.fill_(0.01)

    def forward(self, img, c, **block_kwargs):
        x = self.blocks(img)
        pixel_num = x.shape[2] * x.shape[3]
        channel_num = x.shape[1]
        # channel last arrangement
        x = x.permute(0, 2, 3, 1).reshape((-1, pixel_num*channel_num))
        # print(x.size())
        x = self.fc(x)
        # print(x.size())
        self.mean = self.mean_linear(x)
        self.logvar = self.logvar_linear(x)

        # reparameterized trick
        # print("mean before repraram", self.mean)
        z_sample = self.reparameter(self.mean, self.logvar)  # (N, latent_dim)
        return z_sample, self.mean, self.logvar

    def reparameter(self, mean, logvar):
        """reparameter and return a sampled feature veector"""
        # reparameterization trick for VAE
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps * std + mean
        return z


#----------------------------------------------------------------------------
#################################################
#        Generator with Attention     #
#################################################
# utils
@torch.no_grad()
def concat_all_gather(x):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    # tensors_gather = [torch.ones_like(tensor)
    #     for _ in range(torch.distributed.get_world_size())]
    # torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    x = x.contiguous()
    out_list = [torch.zeros_like(x, device=x.device, dtype=x.dtype)
                for _ in range(dist.get_world_size())]
    dist.all_gather(out_list, x, async_op=False)

    return torch.cat(out_list, dim=0)

def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = dist.get_world_size()
    if world_size == 1:
        return [data]

    # obtain Tensor size of each rank
    local_size = torch.LongTensor([data.shape[0]]).to(data.device)
    size_list = [torch.LongTensor([0]).to(data.device) for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.zeros(size=(max_size, data.shape[1])).to(data.device))

    if local_size != max_size:
        padding = torch.zeros(size=(max_size - local_size, data.shape[1])).to(data.device)
        tensor = torch.cat((data, padding), dim=0)
    else:
        tensor = data

    dist.all_gather(tensor_list, tensor)
    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor[:size]
        data_list.append(buffer)

    return data_list


@persistence.persistent_class
class SelfAttention(torch.nn.Module):
    def __init__(self,
                 feature_ch,                    # the channels of incoming feature map
                 which_attention='linear',      # which attention to use
                 ):
        super().__init__()
        self.feature_ch = feature_ch
        self.latent_ch = feature_ch // 8
        self.attention = which_attention

        which_conv = Conv2dLayer
        self.theta = which_conv(self.feature_ch, self.latent_ch, kernel_size=1, bias=False)
        self.phi = which_conv(self.feature_ch, self.latent_ch, kernel_size=1, bias=False)
        self.g = which_conv(self.feature_ch, self.latent_ch, kernel_size=1, bias=False)
        self.o = which_conv(self.latent_ch, self.feature_ch, kernel_size=1, bias=False)
        # Learnable gain parameter
        self.gamma = torch.nn.Parameter(torch.tensor(0.), requires_grad=True)

    def linear_attention(self, q, k, v):
        q = torch.softmax(q, dim=-2)
        k = torch.softmax(k, dim=-2)
        out = torch.bmm(q.transpose(1, 2), torch.bmm(k, v.transpose(1, 2)))
        return out

    def casual_attention(self, q, k, v):
        # q [batch_size, channel, feature_map]
        attention_score = torch.bmm(q.transpose(1, 2), k)
        attention_score = torch.softmax(attention_score, dim=-1)
        out = torch.bmm(attention_score, v.transpose(1, 2))
        return out

    def forward(self, feature_map):
        # Apply convs
        theta = self.theta(feature_map)   # (N, latent_ch, feature_map_size, feature_map_size)
        phi = self.phi(feature_map)       # (emb_size, latent_ch)
        g = self.g(feature_map)           # (emb_size, latent_ch)

        # Perform reshapes
        theta = theta.view(-1, self.latent_ch, feature_map.shape[2] * feature_map.shape[3])
        phi = phi.view(-1, self.latent_ch, feature_map.shape[2] * feature_map.shape[3])
        g = g.view(-1, self.latent_ch, feature_map.shape[2] * feature_map.shape[3])

        # get attention maps
        if self.attention == 'linear':
            latent_o = self.linear_attention(theta, phi, g).view(-1, self.latent_ch, feature_map.shape[2],
                                                                 feature_map.shape[3])
        else:
            latent_o = self.casual_attention(theta, phi, g).view(-1, self.latent_ch, feature_map.shape[2],
                                                                 feature_map.shape[3])
        # Attention map times g path
        o = self.o(latent_o)
        return self.gamma * o + feature_map


# here is an attention module to be used by memory bank
@persistence.persistent_class
class ConceptAttention(torch.nn.Module):
    def __init__(self,
                 feature_ch,                    # the channels of incoming feature map
                 emb_ch,                        # the channels of vector embedding in the dictionary
                 which_attention='linear',  # which attention to use
                 ):
        super().__init__()
        # Encoder
        self.feature_ch = feature_ch
        self.emb_ch = emb_ch
        self.latent_ch = feature_ch // 8

        self.latent_ch = emb_ch
        self.attention = which_attention

        which_conv = Conv2dLayer
        self.theta = which_conv(self.feature_ch, self.latent_ch, kernel_size=1, bias=False)

        self.phi = FullyConnectedLayer(self.emb_ch, self.latent_ch)
        self.g = FullyConnectedLayer(self.emb_ch, self.latent_ch)

        self.o = which_conv(self.latent_ch, self.feature_ch, kernel_size=1, bias=False)
        # Learnable gain parameter
        self.gamma = torch.nn.Parameter(torch.tensor(0.), requires_grad=True)

    def linear_attention(self, q, k, v):
        q = torch.softmax(q, dim=-2)
        k = torch.softmax(k, dim=-2)
        out = torch.bmm(q.transpose(1, 2), torch.bmm(k, v.transpose(1, 2)))
        return out

    def casual_attention(self, q, k, v):
        # q [batch_size, channel, feature_map]
        attention_score = torch.bmm(q.transpose(1, 2), k)
        attention_score = torch.softmax(attention_score, dim=-1)
        out = torch.bmm(attention_score, v.transpose(1, 2))
        return out

    def forward(self, feature_map, embedding):
        # original embedding should with size (number of embeddings, channels)
        # Apply convs
        theta = self.theta(feature_map)   # (N, latent_ch, feature_map_size, feature_map_size)
        phi = self.phi(embedding)       # (emb_size, latent_ch)
        g = self.g(embedding)           # (emb_size, latent_ch)

        # Perform reshapes
        theta = theta.view(-1, self.latent_ch, feature_map.shape[2] * feature_map.shape[3])

        phi = phi.permute(1, 0)  # channel first
        phi = phi.view(1, self.latent_ch, embedding.shape[0])
        phi = phi.repeat(theta.size(0), 1, 1)

        g = g.permute(1, 0)  # channel first
        g = g.view(1, self.latent_ch, embedding.shape[0])
        g = g.repeat(theta.size(0), 1, 1)

        # get attention maps
        if self.attention == 'linear':
            latent_o = self.linear_attention(theta, phi, g).view(-1, self.latent_ch, feature_map.shape[2],
                                                                 feature_map.shape[3])
        else:
            latent_o = self.casual_attention(theta, phi, g).view(-1, self.latent_ch, feature_map.shape[2],
                                                                 feature_map.shape[3])
        # Attention map times g path
        o = self.o(latent_o)
        return o


# real moca
@persistence.persistent_class
class MomentumConceptAttention(torch.nn.Module):
    def __init__(self,
                 feature_ch,  # the channels of incoming feature map
                 emb_ch,  # the channels of vector embedding in the dictionary
                 which_attention='linear',  # which attention to use
                 m=0.999,            # momentum term
                 trainable_conv=True, # use same encoder, decoder as self-attention?
                 sa_module=None,    # copy parameters from sa Module
                 ):
        super().__init__()
        # Encoder
        self.feature_ch = feature_ch
        self.emb_ch = emb_ch
        self.latent_ch = feature_ch // 8
        assert self.emb_ch == self.latent_ch

        self.attention = which_attention
        self.m = m

        which_conv = Conv2dLayer
        if trainable_conv:
            self.theta = which_conv(self.feature_ch, self.latent_ch, kernel_size=1, bias=False)
        else:
            if sa_module is None:
                raise Exception('cannot create MOCA under non-trainable mode without a SA module')
            self.theta = sa_module.theta
        # phi can only update with momentum
        self.phi = which_conv(self.feature_ch, self.latent_ch, kernel_size=1, bias=False, trainable=False)
        # initialization
        for param_q, param_k in zip(self.theta.parameters(), self.phi.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        if trainable_conv:
            self.o = which_conv(self.latent_ch, self.feature_ch, kernel_size=1, bias=False)
        else:
            if sa_module is None:
                raise Exception('cannot create MOCA under non-trainable mode without a SA module')
            self.o = sa_module.o

        # Learnable gain parameter
        # self.gamma = torch.nn.Parameter(torch.tensor(0.), requires_grad=True)

    def linear_attention(self, q, k, v):
        q = torch.softmax(q, dim=-2)
        k = torch.softmax(k, dim=-2)
        out = torch.bmm(q.transpose(1, 2), torch.bmm(k, v.transpose(1, 2)))
        return out

    def casual_attention(self, q, k, v):
        # q [batch_size, channel, feature_map]
        attention_score = torch.bmm(q.transpose(1, 2), k)
        attention_score = torch.softmax(attention_score, dim=-1)
        out = torch.bmm(attention_score, v.transpose(1, 2))
        return out

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.theta.parameters(), self.phi.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def forward(self, encoded_feature_map, concepts):
        # original embedding should with size (number of embeddings, channels)
        # Apply convs
        theta = encoded_feature_map  # (N, latent_ch, feature_map_size(h), feature_map_size(w))  also can be seen as query
        with torch.no_grad():
            phi = concepts.permute(1, 0).reshape(1, self.latent_ch, concepts.shape[0])   # channel first
            # keys = self.phi(feature_map)      # (N, latent_ch, h, w)   # keys used to update memory

        # Perform reshapes
        theta = theta.view(-1, self.latent_ch, encoded_feature_map.shape[2] * encoded_feature_map.shape[3])
        phi = phi.repeat(theta.size(0), 1, 1)

        g = phi

        # get attention maps
        if self.attention == 'linear':
            latent_o = self.linear_attention(theta, phi, g).view(-1, self.latent_ch,
                                                                 encoded_feature_map.shape[2],
                                                                 encoded_feature_map.shape[3])
        else:
            latent_o = self.casual_attention(theta, phi, g).view(-1, self.latent_ch,
                                                                 encoded_feature_map.shape[2],
                                                                 encoded_feature_map.shape[3])
        # decode
        o = self.o(latent_o)

        return o


@persistence.persistent_class
class ConceptPoolProto(torch.nn.Module):
    random_sample_num = 32
    def __init__(self,
                 feature_ch,
                 emb_ch,
                 cluster_size=100,                     # number of embedding inside cluster
                 which_attention='linear',             # which attention to use
                 momentum=0.999,
                 ):
        super().__init__()

        self.n_embed = cluster_size
        self.emb_ch = emb_ch
        embeddings = torch.randn(cluster_size, emb_ch)

        self.register_buffer('concept_pool', embeddings)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.float32))
        self.attention_module = MomentumConceptAttention(feature_ch, emb_ch, which_attention, m=momentum)

    @torch.no_grad()
    def get_prototype(self):
        emb = self.concept_pool.clone().detach()
        return torch.mean(emb, dim=0, keepdim=True)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr[0])
        move_ptr_back = None
        if self.n_embed % batch_size != 0:  # for simplicity
            while self.n_embed % batch_size != 0:
                batch_size -= 1

        if ptr + batch_size >= self.n_embed:
            batch_size = self.n_embed - 1 - ptr
            move_ptr_back = True
        keys = keys[:batch_size]

        # replace the keys at ptr (dequeue and enqueue)

        self.concept_pool[ptr:ptr + batch_size, :] = keys
        if move_ptr_back is None:
            ptr = (ptr + batch_size) % self.n_embed # move pointer
        else:
            ptr = 0
        self.queue_ptr[0] = ptr

    def forward(self, x: torch.Tensor, sync=False):
        # input_dim [batch_size, feature map channels, h, w]
        if not self.use_momentum:
            out = self.attention_module(x, self.concept_pool)
        else:
            if sync:
                sample_x = x.clone().detach()
                sampled_feature = sample_x.permute(0, 2, 3, 1).reshape(-1, self.emb_ch)
                sample_idx = torch.randperm(sampled_feature.shape[0])[:self.random_sample_num]
                sampled_feature = sampled_feature[sample_idx]
                self._dequeue_and_enqueue(sampled_feature)

            out = self.attention_module(x, self.concept_pool)

        return out


@persistence.persistent_class
class VanillaMoCA(torch.nn.Module):
    # no multiple cluster
    def __init__(self,
                 feature_ch,
                 emb_ch,
                 concept_pool_size=512,  # number of embedding inside concept pool
                 which_attention='linear',  # which attention to use
                 momentum=0.999
                 ):
        super().__init__()
        self.concept_pool = ConceptPoolProto(feature_ch, emb_ch, concept_pool_size, which_attention, momentum)
        self.sa_layer = SelfAttention(feature_ch, which_attention)

    def forward(self, x, sync=False):
        x = self.sa_layer(x)
        x = self.concept_pool(x, sync=sync)
        return x


@persistence.persistent_class
class MoCA(torch.nn.Module):
    warmup_iter = 20000
    rerouting_iter = 20000
    def __init__(self,
                 feature_ch,
                 emb_ch,
                 cluster_num=20,            # Number of cluster
                 concept_pool_size=256,  # number of embedding inside concept pool
                 which_attention='casual',  # which attention to use
                 momentum=0.999,
                 use_sa=True,
                 rerouting=False,
                 preheat=True,
                 ):
        super().__init__()
        self.cluster_num = cluster_num
        self.cluster_size = concept_pool_size
        self.random_sample_num = concept_pool_size // 8

        self.feature_ch = feature_ch
        self.emb_ch = emb_ch
        self.pool_size = {i: concept_pool_size for i in range(cluster_num)}
        self.pool_size['init'] = cluster_num*100

        # zero init
        concept_pool = torch.zeros(cluster_num, concept_pool_size, self.emb_ch)
        self.global_pool = torch.nn.Parameter(concept_pool, requires_grad=False)

        # store keys generated in warmup stage
        self.register_buffer('pool_init', torch.randn(cluster_num*100, self.emb_ch))

        self.use_sa = use_sa
        if self.use_sa:
            self.sa_layer = SelfAttention(feature_ch, which_attention)

        # use own trainable conv
        # self.moca_layer = MomentumConceptAttention(self.feature_ch, self.emb_ch,
        #                                            which_attention=which_attention, m=momentum)
        # # use encoder decoder from SA
        self.moca_layer = MomentumConceptAttention(self.feature_ch, self.emb_ch, m=momentum,
                                                   which_attention=which_attention,
                                                   trainable_conv=False, sa_module=self.sa_layer)

        self.gamma = torch.nn.Parameter(torch.tensor(0.), requires_grad=True)
        # self.sa_norm = torch.nn.InstanceNorm2d(self.feature_ch)
        # self.moca_norm = torch.nn.InstanceNorm2d(self.feature_ch)

        self.register_buffer('init_ptr', torch.zeros(1, dtype=torch.float32))
        self.register_buffer('warmup_count', torch.zeros(1, dtype=torch.float32))
        self.register_buffer("queue_ptr", torch.zeros(cluster_num, dtype=torch.float32))
        self.register_buffer('rerouting_count', torch.zeros(1, dtype=torch.float32))
        if preheat:
            self.already_warmup = False
        else:
            self.already_warmup = True

        self.rerouting = rerouting

    @torch.no_grad()
    def kmeans_init_pool(self):
        print('performing kmeans clustering')
        local_device = self.pool_init.device
        concepts_init = torch.zeros((self.cluster_num, self.cluster_size, self.emb_ch), device=local_device)
        x = self.pool_init.clone().cpu().numpy()
        x = np.ascontiguousarray(x)
        sigma = np.mean(np.abs(x))
        kmeans = KMeans(n_clusters=self.cluster_num, random_state=0).fit(x)

        centroids = kmeans.cluster_centers_
        for p in range(self.cluster_num):
            pool_center = centroids[p]
            concepts_init[p] = torch.randn(self.cluster_size, self.emb_ch).to(local_device) * sigma * 0.5 +\
                               torch.from_numpy(pool_center).to(local_device)

        print("Finish kmean init...")
        return concepts_init

    @torch.no_grad()
    def kmeans_rerouting(self):
        print('performing rerouting')
        local_device = self.pool_init.device
        x = self.global_pool.detach().clone().view(-1, self.emb_ch).cpu().numpy()
        new_pool = self.global_pool.detach().clone()
        x = np.ascontiguousarray(x)
        sigma = np.mean(np.abs(x))

        kmeans = KMeans(n_clusters=self.cluster_num, random_state=0).fit(x)
        label_arr = kmeans.labels_   # same length as x.shape[0]
        centroids = kmeans.cluster_centers_

        # start rerouting
        for p in range(self.cluster_num):
            pool_center = centroids[p]
            # concepts that are in that cluster
            concept_idx = np.argwhere(label_arr == p).reshape(-1)
            if len(concept_idx) > self.cluster_size:
                concept_idx = concept_idx[:self.cluster_size]
            new_pool[p, :, :] = torch.randn_like(self.global_pool[p, :, :]) * sigma * 0.1 +\
                                        torch.from_numpy(pool_center).to(local_device)
            new_pool[p, :concept_idx.shape[0], :] = torch.from_numpy(x[concept_idx]).to(local_device)

        print("Finish rerouting...")
        return new_pool

    @torch.no_grad()
    def gather_prototype(self):
        protos = torch.mean(self.global_pool, dim=0)
        return protos

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, pool_idx):
        # gather keys before updating queue
        keys = torch.cat(all_gather(keys), dim=0)

        batch_size = keys.shape[0]
        if batch_size == 0:
            return

        if not isinstance(pool_idx, str):
            ptr = int(self.queue_ptr[pool_idx])
        else:
            ptr = int(self.init_ptr[0])

        move_ptr_back = None
        n_embed = self.pool_size[pool_idx]  # how many embeddings in this pool?
        if n_embed % batch_size != 0:  # for simplicity
            while n_embed % batch_size != 0:
                batch_size -= 1

        if batch_size == 0:
            return

        if ptr + batch_size >= n_embed:
            batch_size = n_embed - 1 - ptr
            move_ptr_back = True
        keys = keys[:batch_size]
        #print(keys.size())

        # replace the keys at ptr (dequeue and enqueue)
        if not isinstance(pool_idx, str):
            concept_pool = self.global_pool
            concept_pool[pool_idx, ptr:ptr + batch_size, :] = keys.detach()

        else:
            concept_pool = self.pool_init
            concept_pool[ptr:ptr + batch_size, :] = keys.detach()
        #print(concept_pool.size())

        if move_ptr_back is None:
            ptr = (ptr + batch_size) % n_embed  # move pointer
        else:
            ptr = 0   # move back to starting point

        if not isinstance(pool_idx, str):
            self.queue_ptr[pool_idx] = ptr
        else:
            self.init_ptr[0] = ptr

    @torch.no_grad()
    def get_concept_score(self, x):

        protos = self.gather_prototype()  # [cluster_size, c]
        encode_x = self.moca_layer.theta(x)

        encode_x = encode_x.permute(0, 2, 3, 1).reshape(-1, protos.shape[1])   # [n*h*w, c]
        attention_score = torch.mm(encode_x, protos.T)    # [n*h*w, cluster_size]
        cluster_affinity = torch.softmax(attention_score, dim=-1).max(dim=-1)[1]     # [n*h*w, ]
        which_clusters = cluster_affinity               # [n*h*w, ]

        return which_clusters

    def forward(self, x, sync=False):
        # contextual attention
        if self.use_sa:
            sa_out = self.sa_layer(x)
        else:
            sa_out = x

        # MoCA
        encoded_x = self.moca_layer.theta(x)
        concept_keys = self.moca_layer.phi(x)

        # already warmp up
        if self.already_warmup:
            which_clusters = self.get_concept_score(x)
            moca_out = torch.zeros_like(x)
            keys_lst = []

            # concept attention with memory
            for p in range(self.cluster_num):
                cluster_mask = which_clusters == p
                cluster_mask_x = cluster_mask.reshape(x.shape[0], 1, x.shape[2], x.shape[3])  # [batch_size, 1, h, w]
                cluster_mask_x = cluster_mask_x.repeat(1, x.shape[1], 1, 1)
                cluster_mask_key = cluster_mask.view(-1,)

                if int(torch.sum(cluster_mask).item()) == 0:
                    keys_lst += [torch.empty(0, self.moca_layer.emb_ch, dtype=torch.float32, device=moca_out.device)]
                    continue
                concept_pool = self.global_pool[p]

                concept_out = self.moca_layer.forward(encoded_x, concept_pool)
                concept_out_ = concept_out.clone()
                concept_out_[~cluster_mask_x] = 0.
                moca_out += concept_out_

                if int(torch.sum(cluster_mask_key).item()) < self.random_sample_num:
                    new_concept = concept_keys.clone().permute(0, 2, 3, 1).reshape(-1, self.emb_ch)
                    keys_lst += [new_concept]
                    continue
                # encoded key is in the shape [batch_size, latent_ch, h, w]
                sample_idx = torch.randperm(x.shape[0]*x.shape[2]*x.shape[3])
                sample_idx = sample_idx[cluster_mask_key][: self.random_sample_num]

                new_concept = concept_keys.clone().permute(0, 2, 3, 1).reshape(-1, self.emb_ch)[sample_idx]
                keys_lst += [new_concept]   # after masking  [n_keys, latent_ch]
            self.moca_layer._momentum_update_key_encoder()

            # if not in the training mode, do not update pool
            # also if not in ddp, do not update
            if self.training and sync:
                # update concept pool
                with torch.no_grad():
                    for p in range(self.cluster_num):
                        pool_keys = keys_lst[p]
                        self._dequeue_and_enqueue(pool_keys, p)
                if self.rerouting:
                    self.rerouting_count[0] = self.rerouting_count[0] + 1
                    if self.rerouting_count[0] > self.rerouting_iter:
                        self.rerouting_count[0] = 0
                        if dist.get_rank() == 0:
                            concept_rerouted = self.kmeans_rerouting()
                            print('Warmup end...')
                        else:
                            concept_rerouted = torch.zeros_like(self.global_pool)
                        dist.broadcast(concept_rerouted, 0)
                        self.global_pool.data = concept_rerouted.clone().detach()
                        del concept_rerouted # conserve memory
        # haven't warmup
        else:
            concept_pool = self.pool_init
            moca_out = self.moca_layer.forward(encoded_x, concept_pool)
            encoded_key = concept_keys.clone().permute(0, 2, 3, 1).reshape(-1, self.emb_ch)

            sample_idx = torch.randperm(encoded_key.shape[0])[:self.random_sample_num]
            encoded_key = encoded_key[sample_idx]

            self.moca_layer._momentum_update_key_encoder()

            if self.training and sync:
                with torch.no_grad():
                    self._dequeue_and_enqueue(encoded_key, 'init')
                    self.warmup_count[0] = self.warmup_count[0]+1
                    if self.warmup_count[0] > self.warmup_iter and not self.already_warmup:
                        self.already_warmup = True
                        if dist.get_rank() == 0:
                            concept_init = self.kmeans_init_pool()
                            print('Warmup end...')
                        else:
                            concept_init = torch.zeros_like(self.global_pool)
                        dist.broadcast(concept_init, 0)
                        self.global_pool.data = concept_init.clone().detach()
        out = sa_out + moca_out * self.gamma
        return out


@persistence.persistent_class
class MoCASynthesisNetwork(torch.nn.Module):
    def __init__(self,
        atten_resolution,           # on which resolution we apply attention
        momentum,                   # can use this to decide use momentum or not
        pool_size,                  # size of concept pool
        emb_channels,               # Channels of embedding in the concept pool
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output image resolution.
        img_channels,               # Number of color channels.
        cluster_num     = 1,        # Number of cluster
        use_sa          = True,     # whether to use self-attention or not
        rerouting       = False,    # rerouting during training?
        preheat         = True,    # keamns init?
        channel_base    = 32768,    # Overall multiplier for the number of channels.
        channel_max     = 512,      # Maximum number of channels in any layer.
        num_fp16_res    = 0,        # Use FP16 for the N highest resolutions.
        **block_kwargs,             # Arguments for SynthesisBlock.
    ):
        assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0
        super().__init__()
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(2, self.img_resolution_log2 + 1)]
        self.atten_resolution = atten_resolution

        assert self.atten_resolution in self.block_resolutions
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        self.num_ws = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res // 2] if res > 4 else 0
            out_channels = channels_dict[res]
            use_fp16 = (res >= fp16_resolution)
            is_last = (res == self.img_resolution)
            block = SynthesisBlock(in_channels, out_channels, w_dim=w_dim, resolution=res,
                img_channels=img_channels, is_last=is_last, use_fp16=use_fp16, **block_kwargs)

            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb
            setattr(self, f'b{res}', block)
            if res == self.atten_resolution:
                if cluster_num == 1:                    # no cluster
                    atten_block = VanillaMoCA(out_channels, out_channels//8,
                                              concept_pool_size=pool_size, momentum=momentum)
                else:                                   # with cluster
                    atten_block = MoCA(out_channels, out_channels//8,
                                       cluster_num=cluster_num,
                                       concept_pool_size=pool_size,
                                       momentum=momentum,
                                       use_sa=use_sa,
                                       rerouting=rerouting,
                                       preheat=preheat)

                setattr(self, f'atten{res}', atten_block)


    def forward(self, ws, sync=False, **block_kwargs):
        block_ws = []
        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
            ws = ws.to(torch.float32)
            w_idx = 0
            for res in self.block_resolutions:
                block = getattr(self, f'b{res}')
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv

        x = img = None
        for res, cur_ws in zip(self.block_resolutions, block_ws):
            block = getattr(self, f'b{res}')
            x, img = block(x, img, cur_ws, **block_kwargs)
            if res == self.atten_resolution:
                x = x.to(torch.float32)
                atten_block = getattr(self, f'atten{res}')
                x = atten_block(x, sync)

        return img

# same as original generator
# but with moca
@persistence.persistent_class
class MoCAGenerator(torch.nn.Module):
    def __init__(self,
                 emb_channels,              # Channels of embedding in the concept pool
                 momentum,                  # use momentum or not
                 pool_size,                 # size of pool
                 z_dim,                     # Input latent (Z) dimensionality.
                 c_dim,                     # Conditioning label (C) dimensionality.
                 w_dim,                     # Intermediate latent (W) dimensionality.
                 img_resolution,            # Output resolution.
                 img_channels,              # Number of output color channels.
                 cluster_num=1,             # Number of cluster to be used in MoCA
                 low_level_moca=False,      # add moca to lower level
                 use_sa=True,               # whether to use self-attention or not
                 rerouting=False,           # rerouting during training?
                 preheat=True,             # preheat
                 mapping_kwargs={},         # Arguments for MappingNetwork.
                 synthesis_kwargs={},       # Arguments for SynthesisNetwork.
                 ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        if low_level_moca:
            atten_resolution = int(img_resolution // 4) if img_resolution < 128 else int(img_resolution // 8)
        else:
            atten_resolution = int(img_resolution // 2)
        self.synthesis = MoCASynthesisNetwork(atten_resolution=atten_resolution,
                                              momentum=momentum,
                                              pool_size=pool_size,
                                              emb_channels=emb_channels,
                                              use_sa=use_sa,
                                              cluster_num=cluster_num,
                                              rerouting=rerouting,
                                              preheat=preheat,
                                              w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels,
                                              **synthesis_kwargs)
        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)

    def forward(self, z, c, truncation_psi=1,
                truncation_cutoff=None, sync=False, **synthesis_kwargs):
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        img = self.synthesis(ws, sync, **synthesis_kwargs)
        return img


# MoCA discriminator
@persistence.persistent_class
class MoCADiscriminator(torch.nn.Module):
    def __init__(self,
                emb_channels,                   # Channels of embedding in the concept pool
                momentum,                       # use momentum or not
                pool_size,                      # size of pool
                c_dim,                          # Conditioning label (C) dimensionality.
                img_resolution,                 # Input resolution.
                img_channels,                   # Number of input color channels.
                cluster_num = 1,                # Number of cluster to be used in MoCA
                architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
                channel_base        = 32768,    # Overall multiplier for the number of channels.
                channel_max         = 512,      # Maximum number of channels in any layer.
                num_fp16_res        = 0,        # Use FP16 for the N highest resolutions.
                conv_clamp          = None,     # Clamp the output of convolution layers to +-X, None = disable clamping.
                cmap_dim            = None,     # Dimensionality of mapped conditioning label, None = default.
                block_kwargs        = {},       # Arguments for DiscriminatorBlock.
                mapping_kwargs      = {},       # Arguments for MappingNetwork.
                epilogue_kwargs     = {},       # Arguments for DiscriminatorEpilogue.
    ):
        super().__init__()
        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        # symmetrically add moca
        self.atten_resolution = int(img_resolution // 2)

        if cmap_dim is None:
            cmap_dim = channels_dict[4]
        if c_dim == 0:
            cmap_dim = 0

        common_kwargs = dict(img_channels=img_channels, architecture=architecture, conv_clamp=conv_clamp)
        cur_layer_idx = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res] if res < img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            use_fp16 = (res >= fp16_resolution)
            block = DiscriminatorBlock(in_channels, tmp_channels, out_channels, resolution=res,
                first_layer_idx=cur_layer_idx, use_fp16=use_fp16, **block_kwargs, **common_kwargs)
            setattr(self, f'b{res}', block)
            cur_layer_idx += block.num_layers

            # moca block
            if res == self.atten_resolution:
                if cluster_num == 1:                    # no cluster
                    atten_block = VanillaMoCA(out_channels, emb_channels,
                                              concept_pool_size=pool_size, momentum=momentum)
                else:                                   # with cluster
                    atten_block = MoCA(out_channels, emb_channels,
                                       cluster_num=cluster_num,
                                       concept_pool_size=pool_size, momentum=momentum)

                setattr(self, f'atten{res}', atten_block)
        if c_dim > 0:
            self.mapping = MappingNetwork(z_dim=0, c_dim=c_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None, **mapping_kwargs)
        self.b4 = DiscriminatorEpilogue(channels_dict[4], cmap_dim=cmap_dim, resolution=4, **epilogue_kwargs, **common_kwargs)

    def forward(self, img, c, sync=False, **block_kwargs):
        x = None
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            x, img = block(x, img, **block_kwargs)
            if res == self.atten_resolution:
                x = x.to(torch.float32)
                atten_block = getattr(self, f'atten{res}')
                x = atten_block(x, sync)

        cmap = None
        if self.c_dim > 0:
            cmap = self.mapping(None, c)
        x = self.b4(x, img, cmap)
        return x


if __name__ == '__main__':
    # some simple testing
    fake_feature_map = torch.randn((16, 512, 16, 16))
    concept_cluster = ConceptPoolProto(512, 128, 100)
    c = concept_cluster.forward(fake_feature_map)
    sa = SelfAttention(512)
    s = sa.forward(c)
    print(c.size())
    print(s.size())

    moca = VanillaMoCA(512, 128, 512)
    m = moca.forward(fake_feature_map)
    print(m.size())
    print(s-m)