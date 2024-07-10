# This is the code for the paper I2IDifT
Here are some suggestions I would like to make (if you want to reproduce I2IDiffT).

Firstly, you need to train a very good encoder and decoder in the VAE framework, which is crucial for I2IDift. Because I2IDifT requires it to achieve mutual mapping between the original image space and the latent representation space. 
When training VAE on IXI and BRATS datasets, we strongly recommend using a tiny structure of VAE (we provide this version of the code: '/testmodel/Improvedvae_tiny.py').   
During the training process, please use GAN loss with caution. Although it can improve the performance of the model, it is very prone to pattern collapse. So during the training process, please pay attention to the following points:
*    Normalize the inputs (very important!!!)
*    Please normalize the images between -1 and 1 and Tanh as the last layer of the generator output (very important!!!)
*    Do not use the std and mean values of original data, because this will causing data information loss when return to original image space.
*    the stability of the GAN game suffers if you have sparse gradients
*    LeakyReLU = good (in both G and D)
*    For Downsampling, use: Average Pooling, Conv2d + stride
*    For Upsampling, use: PixelShuffle, ConvTranspose2d + stride


When train the denoising diffusion modules, we highly recommend you to convert the source contrast data and target contrast data into tensor form in advance for storage, which can save a lot of data loading time.

In addition, we strongly recommend that you calculate the variance of the data in the latent space in advance. You may think that the variance should be 1, as we set the corresponding KL loss when training VAE. 
But if you check the variance of the data when training the diffusion modules, 
you will find that the variance is not equal to 1, and you need to multiply it by a number to standardize the data. Of course, when decoding again, you need to restore the data.
