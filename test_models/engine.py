import time
import math
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
torch.backends.cudnn.benchmark = True


def get_lr_basic(it, config):
    if it < config.warmup_iters:
        return config.learning_rate * it / config.warmup_iters
    if it > config.lr_decay_iters:
        return config.min_lr
    decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


train_tensor = torch.load('./data/train_f_tensor.pt')
dataset = torch.utils.data.TensorDataset(train_tensor)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

valid_path = './core_data/valid_f_norm.pt'
valid_data = torch.load(valid_path)[:128]
valid_real = torch.load('./core_data/valid_real.pt')[:128]

def train(train_loader, model, optimizer, scaler, config, iter_num, get_lr):
    rec_loss_knt = 0
    kl_loss_knt = 0
    pl_loss_knt = 0
    loss_knt = 0
    g_loss_knt = 0
    d_loss_knt = 0
    d_weight_knt = 0
    loss_num_g = 0
    loss_num_d = 0

    model.train()
    st = time.time()
    for opt in optimizer:
        opt.zero_grad(set_to_none=True)
    for i, data in enumerate(train_loader):
        lr = get_lr(iter_num)
        need_g_loss = iter_num > config.disc_start
        optimizer_idx = iter_num % 2 if need_g_loss else 0  #####
        opt = optimizer[optimizer_idx]
        for param_group in opt.param_groups:
            param_group['lr'] = lr
        input = data[0].to('cuda', non_blocking=True)
        with autocast(dtype=torch.bfloat16):
            rec_loss, kl_loss, pl_loss, loss, d_weight, g_loss, d_loss = model(input, optimizer_idx=optimizer_idx,
                                                                               need_g_loss=need_g_loss)

        if optimizer_idx == 0:
            loss /= config.gradient_accumulation_steps
            scaler.scale(loss).backward()
            loss_knt += loss.item() / config.gradient_accumulation_steps
            rec_loss_knt += rec_loss.item() / config.gradient_accumulation_steps
            kl_loss_knt += kl_loss.item() / config.gradient_accumulation_steps
            pl_loss_knt += pl_loss.item() / config.gradient_accumulation_steps
            g_loss_knt += g_loss.item() / config.gradient_accumulation_steps
            d_weight_knt += d_weight / config.gradient_accumulation_steps
        else:
            d_loss /= config.gradient_accumulation_steps
            scaler.scale(d_loss).backward()
            d_loss_knt += d_loss.item()

        if (i + 1) % config.gradient_accumulation_steps == 0:
            if optimizer_idx == 0:
                loss_num_g += 1
            else:
                loss_num_d += 1
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            iter_num += 1
            if iter_num % 50 == 0:
                if loss_num_g == 0:
                    loss_num_g += 1
                print(f"step {iter_num}, "
                      f"loss {loss_knt / loss_num_g:.3f}, "
                      f"rec {rec_loss_knt / loss_num_g:.3f}, "
                      f"kl {kl_loss_knt / loss_num_g:.3f}, "
                      f"pl {pl_loss_knt / loss_num_g:.3f}, "
                      f"g {g_loss_knt / loss_num_g:.3f}, "
                      f"w {d_weight_knt / loss_num_g:.3f}, "
                      f"d {d_loss_knt / loss_num_d if loss_num_d else 0:.3f}, "
                      f"lr: {lr:.7f}, "
                      f"consume {time.time() - st:.2f}s")
                st = time.time()
                rec_loss_knt = 0
                kl_loss_knt = 0
                pl_loss_knt = 0
                loss_knt = 0
                g_loss_knt = 0
                d_weight_knt = 0
                d_loss_knt = 0
                loss_num_g = 0
                loss_num_d = 0
            if iter_num >= config.max_iters:
                break
    return iter_num