import math
import torch
import torch.nn as nn
from .discriminator import Discriminator, hinge_d_loss
from .lpips import LPIPS
from .vae import Encoder, Decoder

    
class ImprovedVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.final_layer = nn.Tanh() # 先不过这一层
        self.discriminator = Discriminator()
        self.pl_model = LPIPS()
        self.last_layer = self.decoder.conv_out.weight

    def get_vae_params(self):
        return list(self.encoder.parameters())+ list(self.decoder.parameters())

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

    def decode(self, z):
        result = self.decoder(z)
        # result = self.final_layer(result)
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
            rec_loss, kl_loss, pl_loss, g_loss, loss, d_weight = self.loss(input, rec, mu, log_var, need_g_loss, optimizer_idx!=2)
        if optimizer_idx == 1 or optimizer_idx == 2:
            d_loss = self.disc_loss(input, rec)
        return rec_loss, kl_loss, pl_loss, loss, d_weight, g_loss, d_loss
    
    def loss(self, input, rec, mu, log_var, need_g_loss=False, need_d_weight=True):
        rec_loss = nn.functional.l1_loss(rec, input)
        kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=[1, 2, 3]), dim = 0) * 5e-6
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
        d_loss = hinge_d_loss(logits_real, logits_fake)
        return d_loss
    
    def calculate_adaptive_weight(self, rec_loss, g_loss):
        rec_grads = torch.autograd.grad(rec_loss, self.last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, self.last_layer, retain_graph=True)[0]

        d_weight = torch.linalg.norm(rec_grads) / (torch.linalg.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight * 0.5
    

# if __name__ == "__main__":
#     from dataclasses import dataclass
#     @dataclass
#     class ImprovedVAE_Config:
#         layers = [3, 3, 3, 3]
#         channels = [64, 64, 128, 256, 256]
#         z_channel = 8
#         kl_weight = 0.00025

#     x = ImprovedVAE(config)
#     print(x)
    # print(x.get_num_params())
    # y = torch.zeros([10,3,64,64])
    # z = x(y)