import numpy as np
import torch
import torch.nn as nn
from diffusers.models.attention_processor import Attention
from typing import Optional, Callable, Any, Tuple, Union, List
import torch.nn.functional as F
from ml_collections import config_dict
from .discriminator import Discriminator, hinge_d_loss
from .lpips import LPIPS

def med_vae_config():
    config = config_dict.ConfigDict()
    config.encoder = {
        'in_channels': 1,
        'out_channels': 2,
        'block_out_channels': (32, 64, 128, 128),
        'layers_per_block': 1,
        'norm_num_groups': 16,
        'double_z': True
    }

    config.decoder = {
        'in_channels': 2,
        'out_channels': 1,
        'block_out_channels': (32, 64, 128, 128),
        'layers_per_block': 1,
        'norm_num_groups': 16,
        'output_size': 224
    }
    return config

config = med_vae_config()


class Downsample2D(nn.Module):

    def __init__(
            self,
            channels: int,
            out_channels: Optional[int] = None,
            padding: int = 1,
            kernel_size=3,
            bias=True,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.padding = padding
        stride = 2

        self.conv = nn.Conv2d(
            self.channels, self.out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias
        )

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        assert hidden_states.shape[1] == self.channels
        hidden_states = self.conv(hidden_states)

        return hidden_states


class DownEncoderBlock2D(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            dropout: float = 0.0,
            num_layers: int = 2,
            resnet_groups: int = 16,
            add_downsample: bool = True,
            downsample_padding: int = 1,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    groups=resnet_groups,
                    dropout=dropout,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels, out_channels=out_channels, padding=downsample_padding,
                    )
                ]
            )
        else:
            self.downsamplers = None

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=None)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

        return hidden_states


class ResnetBlock2D(nn.Module):

    def __init__(
            self,
            in_channels: int,  ###
            out_channels: Optional[int] = None,  ###
            conv_shortcut: bool = False,
            dropout: float = 0.0,  ###
            groups: int = 16,  ####
            groups_out: Optional[int] = None,
            eps: float = 1e-6,  ####
            output_scale_factor: float = 1.0,  ####
            use_in_shortcut: Optional[bool] = None,
            conv_shortcut_bias: bool = True,
            conv_2d_out_channels: Optional[int] = None,
    ):
        super().__init__()

        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.output_scale_factor = output_scale_factor

        if groups_out is None:
            groups_out = groups

        self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = torch.nn.GroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True)
        self.dropout = torch.nn.Dropout(dropout)
        conv_2d_out_channels = conv_2d_out_channels or out_channels
        self.conv2 = nn.Conv2d(out_channels, conv_2d_out_channels, kernel_size=3, stride=1, padding=1)
        self.nonlinearity = nn.SiLU()
        self.use_in_shortcut = self.in_channels != conv_2d_out_channels if use_in_shortcut is None else use_in_shortcut
        self.conv_shortcut = None

        if self.use_in_shortcut:
            self.conv_shortcut = nn.Conv2d(
                in_channels,
                conv_2d_out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=conv_shortcut_bias,
            )

    def forward(self, input_tensor: torch.Tensor, temb: torch.Tensor, *args, **kwargs) -> torch.Tensor:

        hidden_states = input_tensor
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor


class UNetMidBlock2D(nn.Module):

    def __init__(
            self,
            in_channels: int,
            dropout: float = 0.0,
            resnet_eps: float = 1e-6,
            resnet_groups: int = 32,
            output_scale_factor: float = 1.0,
    ):
        super().__init__()
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        attn_groups = resnet_groups

        resnets = [
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                groups=resnet_groups,
                dropout=dropout,
            )
        ]
        attentions = []
        attention_head_dim = in_channels
        attentions.append(
            Attention(
                in_channels,
                heads=in_channels // attention_head_dim,
                dim_head=attention_head_dim,
                rescale_output_factor=output_scale_factor,
                eps=resnet_eps,
                norm_num_groups=attn_groups,
                spatial_norm_dim=None,
                residual_connection=True,
                bias=True,
                upcast_softmax=True,
                _from_deprecated_attn_block=True,
            )
        )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        for attn, resnet in zip(self.attentions, self.resnets):
            hidden_states = attn(hidden_states, temb=temb)
            hidden_states = resnet(hidden_states, temb)

        return hidden_states


class Encoder(nn.Module):

    def __init__(
            self,
            in_channels=config.encoder['in_channels'],
            out_channels=config.encoder['out_channels'],
            block_out_channels=config.encoder['block_out_channels'],
            layers_per_block=config.encoder['layers_per_block'],
            norm_num_groups=config.encoder['norm_num_groups'],
            double_z=config.encoder['double_z']
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.down_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i in range(4):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == 3

            down_block = DownEncoderBlock2D(
                num_layers=self.layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=not is_final_block,
                downsample_padding=1,
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
        )

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()

        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = nn.Conv2d(block_out_channels[-1], conv_out_channels, 3, padding=1)

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        r"""The forward method of the `Encoder` class."""

        sample = self.conv_in(sample)

        # down
        for down_block in self.down_blocks:
            sample = down_block(sample)

        # middle
        sample = self.mid_block(sample)

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample


class Upsample2D(nn.Module):

    def __init__(
            self,
            channels: int,
            use_conv: bool = True,
            use_conv_transpose: bool = True,
            out_channels: Optional[int] = None,
            name: str = "conv",
            kernel_size: Optional[int] = None,
            padding=1,
            eps=None,
            bias=True,
            interpolate=False,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name
        self.interpolate = interpolate

        self.norm = None

        if use_conv_transpose:
            if kernel_size is None:
                kernel_size = 4
            self.conv = nn.ConvTranspose2d(
                channels, self.out_channels, kernel_size=kernel_size, stride=2, padding=padding, bias=bias
            )
        elif use_conv:
            if kernel_size is None:
                kernel_size = 3
            self.conv = nn.Conv2d(self.channels, self.out_channels, kernel_size=kernel_size, padding=padding, bias=bias)

    def forward(self, hidden_states: torch.Tensor, output_size: Optional[int] = None, *args, **kwargs) -> torch.Tensor:

        assert hidden_states.shape[1] == self.channels

        if self.use_conv_transpose:
            return self.conv(hidden_states)

        dtype = hidden_states.dtype
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.float32)

        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()

        # if `output_size` is passed we force the interpolation output
        # size and do not make use of `scale_factor=2`
        output_size = config.decoder['output_size']
        if self.interpolate:
            if output_size is None:
                hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
            else:
                hidden_states = F.interpolate(hidden_states, size=output_size, mode="nearest")

        # If the input is bfloat16, we cast back to bfloat16
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(dtype)

        hidden_states = self.conv(hidden_states)

        return hidden_states


class UpDecoderBlock2D(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            resolution_idx: Optional[int] = None,
            dropout: float = 0.0,
            num_layers: int = 1,
            resnet_groups: int = 32,
            resnet_pre_norm: bool = True,
            output_scale_factor: float = 1.0,
            add_upsample: bool = True,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=input_channels,
                    out_channels=out_channels,
                    groups=resnet_groups,
                    dropout=dropout,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

        self.resolution_idx = resolution_idx

    def forward(self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=temb)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states


class Decoder(nn.Module):

    def __init__(
            self,
            in_channels=config.decoder['in_channels'],
            out_channels=config.decoder['out_channels'],
            block_out_channels=config.decoder['block_out_channels'],
            layers_per_block=config.decoder['layers_per_block'],
            norm_num_groups=config.decoder['norm_num_groups'],
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[-1],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.up_blocks = nn.ModuleList([])

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_groups=norm_num_groups,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]

        prev_output_channel = None
        for i in range(4):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            is_final_block = i == 3

            up_block = UpDecoderBlock2D(
                num_layers=self.layers_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                add_upsample=not is_final_block,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out

        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        r"""The forward method of the `Decoder` class."""

        sample = self.conv_in(sample)
        upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
        # middle
        sample = self.mid_block(sample)
        sample = sample.to(upscale_dtype)
        # up
        for up_block in self.up_blocks:
            sample = up_block(sample)
        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample


###########################

class ImprovedVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.final_layer = nn.Tanh()
        self.discriminator = Discriminator()
        self.pl_model = LPIPS()
        self.last_layer = self.decoder.conv_out.weight

    def get_vae_params(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters())

    def get_disc_params(self):
        return self.discriminator.parameters()

    def get_num_params(self):
        encoder_params = self.encoder.get_num_params()
        decoder_params = self.decoder.get_num_params()
        discriminator_params = self.discriminator.get_num_params()
        pl_params = self.pl_model.get_num_params()
        n_params = sum(p.numel() for p in self.parameters())
        return n_params, encoder_params, decoder_params, discriminator_params, pl_params

    def encode(self, input):
        result = self.encoder(input)
        mu, log_var = torch.chunk(result, 2, dim=1)
        log_var = torch.clamp(log_var, -30.0, 20.0)
        return [mu, log_var]

    def reparameterize(self, mu, logvar, sample_posterior=True):
        if sample_posterior == False:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def encode_sample(self, input):
        mu, _ = self.encode(input)
        return mu

    def decode(self, z):
        result = self.decoder(z)
        result = self.final_layer(result)
        return result

    def forward(self, input, optimizer_idx=None, need_g_loss=False, sample_posterior=True):
        # optimizer_idx None: infer, 0: 计算vae的loss，1: 计算disc的loss, 2: valid mode, 都计算
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var, sample_posterior)
        rec = self.decode(z)

        if optimizer_idx is None:
            return rec

        rec_loss, kl_loss, pl_loss, loss, d_weight, g_loss, d_loss = [0] * 7
        if optimizer_idx == 0 or optimizer_idx == 2:
            # rec_loss, kl_loss, pl_loss, g_loss, loss, d_weight = self.loss(input, rec, mu, log_var, need_g_loss,optimizer_idx != 2)
            rec_loss, kl_loss, pl_loss, g_loss, loss, d_weight = self.loss(input, rec, mu, log_var, need_g_loss,True)
        if optimizer_idx == 1 or optimizer_idx == 2:
            d_loss = self.disc_loss(input, rec)
        return rec_loss, kl_loss, pl_loss, loss, d_weight, g_loss, d_loss

    def loss(self, input, rec, mu, log_var, need_g_loss=False, need_d_weight=True):
        rec_loss = 5 * nn.functional.l1_loss(rec, input)  #############################
        kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=[1, 2, 3]), dim=0) * 5e-6
        pl_loss = self.pl_model(input, rec) * 0.1

        g_loss, d_weight = torch.zeros(1).to(kl_loss.device), 1.0
        if need_g_loss:
            g_loss = torch.mean(nn.functional.relu(-self.discriminator(rec)))
            if need_d_weight:
                d_weight = self.calculate_adaptive_weight(rec_loss, g_loss)
            g_loss = g_loss * d_weight

        loss = rec_loss + kl_loss + pl_loss + g_loss
        return rec_loss, kl_loss, pl_loss, g_loss, loss, d_weight

    def disc_loss(self, input, rec):
        logits_real = self.discriminator(input.detach())
        logits_fake = self.discriminator(rec.detach())
        d_loss = hinge_d_loss(logits_real, logits_fake) * 0.1
        return d_loss

    def calculate_adaptive_weight(self, rec_loss, g_loss):
        rec_grads = torch.autograd.grad(rec_loss, self.last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, self.last_layer, retain_graph=True)[0]

        d_weight = torch.linalg.norm(rec_grads) / (torch.linalg.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e+4).detach()
        return d_weight * 0.5 * 2   ##################################

