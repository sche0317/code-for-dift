import torch
from torcheval.metrics.functional import peak_signal_noise_ratio as psnr

T = 1000

betas=torch.linspace(0.0001,0.02,T) # (T,)
alphas=1-betas  # (T,)
alphas_cumprod=torch.cumprod(alphas,dim=-1) # alpha_t累乘 (T,)    [a1,a2,a3,....] ->  [a1,a1*a2,a1*a2*a3,.....]
alphas_cumprod_prev=torch.cat((torch.tensor([1.0]),alphas_cumprod[:-1]),dim=-1) # alpha_t-1累乘 (T,),  [1,a1,a1*a2,a1*a2*a3,.....]
variance=(1-alphas)*(1-alphas_cumprod_prev)/(1-alphas_cumprod)  # denoise用的方差   (T,)

batch_size= 256#######
noise_init=torch.randn(size=(batch_size,2,28,28))

#### t1->pd
condition_path = r'E:\workspace\0个人工作站\深度学习\ML\pythonProject\paper\data\t1_pd\condition_t1.pt'
condition = torch.load(condition_path)
real_path = r'E:\workspace\0个人工作站\深度学习\ML\pythonProject\paper\data\t1_pd\real_pd.pt'
real = torch.load(real_path)
#####

def backward_denoise(model, x=noise_init, y=condition):
    global alphas, alphas_cumprod, variance

    x = x.cuda()
    alphas = alphas.cuda()
    alphas_cumprod = alphas_cumprod.cuda()
    variance = variance.cuda()
    y = y.cuda()

    model.eval()
    with torch.no_grad():
        for time in range(T - 1, -1, -1):
            t = torch.full((x.size(0),), time).cuda()

            # 预测x_t时刻的噪音
            noise = model(x, t, y)

            # 生成t-1时刻的图像
            shape = (x.size(0), 1, 1, 1)
            mean = 1 / torch.sqrt(alphas[t].view(*shape)) * \
                   (
                           x -
                           (1 - alphas[t].view(*shape)) / torch.sqrt(1 - alphas_cumprod[t].view(*shape)) * noise
                   )
            if time != 0:
                x = mean + \
                    torch.randn_like(x) * \
                    torch.sqrt(variance[t].view(*shape))
            else:
                x = mean
            x = torch.clamp(x, -1.0, 1.0).detach()
    return x

def compute_metrics(pred, real=real):
    pred = (pred+1) * 127.5
    pred = torch.clamp(pred, 0, 255)
    real = (real+1) * 127.5
    real = torch.clamp(real, 0, 255)
    real = real.cuda()
    average_psnr = psnr(pred, real)
    print(f"每张灰度图片的PSNR平均值为: {average_psnr} dB")
    return average_psnr

def output(net, if_predict=False):
    pred = backward_denoise(model=net, x=noise_init, y=condition)
    if if_predict:
        return pred

    average_psnr = compute_metrics(pred)
    return average_psnr