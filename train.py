import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from tqdm import tqdm
from test_models.dift_base import DifT_models
from time import time
from ml_collections import config_dict

def main(model, args):
    # Create model:
    model = model
    model = model.cuda()

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    loss_fn = nn.L1Loss()

    # Setup data:
    t1 = torch.load('t1_vae_encode_norm.pt')
    pd = torch.load('pd_vae_encode_norm.pt')
    dataset = TensorDataset(pd, t1)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    print(f"Dataset contains {len(dataset):,} images")

    # Prepare models for training:
    model.train()  # important! This enables embedding dropout for classifier-free guidance

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    print(f"Training for {args.epochs} epochs...")
    pre_loss = args.save_loss
    for epoch in tqdm(range(args.epochs)):
        print(f"Beginning epoch {epoch}...")
        for x, y in loader:
            x = x.cuda()
            y = y.cuda()

            pred = model(x)
            loss = args.loss_weight * loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all steps:
                avg_loss = torch.tensor(running_loss / log_steps, device='cuda')
                print(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

                if avg_loss < pre_loss:
                    pre_loss = avg_loss
                    torch.save(model.state_dict(), 'base_t1_pd_f_loss_{:.4f}.pth'.format(avg_loss))
                    print(f'model has saved, current loss is {avg_loss}')

        model.train()
    print("Done!")

def train_config():
    config = config_dict.ConfigDict()
    config.log_every = 100
    config.epochs = 50    ###############################
    config.batch_size = 16
    config.psnr = 30
    config.loss_weight = 1
    config.condition_weight = 1
    config.lr = 1e-4   ###############################
    config.save_loss = 0.012  ###############################
    return config

if __name__ == "__main__":

    model = DifT_models['I2IDifT_base']()
    args = train_config()
    main(model, args)