import torch
from .Improvedvae_tiny import ImprovedVAE
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.utils.data import DataLoader
from time import time
from .inference import backward_denoise, compute_metrics ### just give a model in cuda is enough
from .diffusion import forward_add_noise
from tqdm import tqdm

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def main(model, args):
    # Create model:
    model = model
    model = model.cuda()
    vae = ImprovedVAE()
    vae.load_state_dict(torch.load("VAE_tiny_f_psnr_33.95.pth"))
    vae.cuda()
    print(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    loss_fn = torch.nn.L1Loss()

    # Setup data:

    train_tensor = torch.load('./data/pd_f_norm.pt')
    conditions = torch.load('./data/t1_f_norm.pt')
    dataset = torch.utils.data.TensorDataset(train_tensor, conditions)
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
            with torch.no_grad():
                # Map input images to latent space
                x = vae.encode_sample(x)
                y = vae.encode_sample(y) * args.condition_weight
            t = torch.randint(0, 1000, (x.shape[0],), device='cuda')

            x, noise = forward_add_noise(x, t)  # x:加噪图 noise:噪音
            pred_noise = model(x, t, y)
            loss = args.loss_weight * loss_fn(pred_noise, noise.cuda())
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

        # model.eval()  # important! This disables randomized embedding dropout
        # with torch.no_grad():
        #     temp = backward_denoise(model)
        #     temp = vae.decode(temp)
        #     avg = compute_metrics(temp)
        #     del temp
        # if avg > args.psnr:
        #     checkpoint_path = 't1_pd_f_psnr_{:.2f}.pth'.format(avg)
        #     torch.save(model.state_dict(), checkpoint_path)
        #     print(f"Saved checkpoint to {checkpoint_path}")

        model.train()
    print("Done!")