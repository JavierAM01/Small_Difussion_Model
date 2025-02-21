import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import wandb

def extract(a, t, x_shape):
    """
    This function abstracts away the tedious indexing that would otherwise have
    to be done to properly compute the diffusion equations from lecture. This
    is necessary because we train data in batches, while the math taught in
    lecture only considers a single sample.
    
    To use this function, consider the example
        alpha_t * x
    To compute this in code, we would write
        extract(alpha, t, x.shape) * x

    Args:
        a: 1D tensor containing the value at each time step.
        t: 1D tensor containing a batch of time indices.
        x_shape: The reference shape.
    Returns:
        The extracted tensor.
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def cosine_schedule(timesteps, s=0.008):
    """
    Passes the input timesteps through the cosine schedule for the diffusion process
    Args:
        timesteps: 1D tensor containing a batch of time indices.
        s: The strength of the schedule.
    Returns:
        1D tensor of the same shape as timesteps, with the computed alpha.
    """
    steps = timesteps + 1
    x = torch.linspace(0, steps, steps)
    alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    return torch.clip(alphas, 0.001, 1)


# normalization functions
def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


# DDPM implementation
class Diffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        channels=3,
        timesteps=1000,
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.model = model
        self.num_timesteps = int(timesteps)

        """
        Initializes the diffusion process.
            1. Setup the schedule for the diffusion process.
            2. Define the coefficients for the diffusion process.
        Args:
            model: The model to use for the diffusion process.
            image_size: The size of the images.
            channels: The number of channels in the images.
            timesteps: The number of timesteps for the diffusion process.
        """
        ## TODO: Implement the initialization of the diffusion process ##
        # 1. define the scheduler here
        # 2. pre-compute the coefficients for the diffusion process
        
        self.device = next(model.parameters()).device
        
        # Define the scheduler
        self.alphas = cosine_schedule(self.num_timesteps).to(self.device)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(self.device)
        
        # Precompute diffusion process coefficients
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(self.device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod).to(self.device)
        
        # ###########################################################

    def noise_like(self, shape, device):
        """
        Generates noise with the same shape as the input.
        Args:
            shape: The shape of the noise.
            device: The device on which to create the noise.
        Returns:
            The generated noise.
        """
        noise = lambda: torch.randn(shape, device=device)
        return noise()

    # backward diffusion
    @torch.no_grad()
    def p_sample(self, x, t, t_index):
        """
        Computes the (t_index)th sample from the (t_index + 1)th sample using
        the reverse diffusion process.
        Args:
            x: The sampled image at timestep t_index + 1.
            t: 1D tensor of the index of the time step.
            t_index: Scalar of the index of the time step.
        Returns:
            The sampled image at timestep t_index.
        """
        ####### TODO: Implement the p_sample function #######
        # sample x_{t-1} from the gaussian distribution wrt. posterior mean and posterior variance
        # Hint: use extract function to get the coefficients at time t
        # Hint: use self.noise_like function to generate noise. DO NOT USE torch.randn
        # Begin code here
        
        # coef_mean = extract(self.sqrt_alphas_cumprod, t, x.shape).to(self.device)
        # coef_var = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape).to(self.device)
        # pred_noise = self.model(x, t)
        # x = x.to(self.device)
        # x_prev = (x - coef_var * pred_noise) / coef_mean
        
        # if t_index == 0:
        #     return x_prev.cpu()
        # else:
        #     noise = self.noise_like(x.shape, self.device)
        #     return (x_prev + noise * coef_var).cpu()

        alpha_t = extract(self.alphas, t, x.shape)  # getting noise schedule at time t
        alpha_t_bar= extract(self.alphas_cumprod ,t,x.shape) # cumulation of alphas until time t
        # alpha_t_bar_prev= extract(torch.cumprod(self.alphas, dim=0),t-1,x.shape)
        if (t_index == 0):
            # alpha_t_bar_prev = extract(torch.cumprod(self.alphas, dim=0), t, x.shape)  # Set to 1 if t=0, as alpha_t_bar_prev is 1 at t=0
            alpha_t_bar_prev = torch.ones_like(alpha_t_bar)
        else:
            alpha_t_bar_prev = extract(self.alphas_cumprod , t-1, x.shape)
         # Predict noise using the model
        epsilon = self.model(x,t)  # predicited noise

        # computing x0
        x_hat_0= (1/torch.sqrt(alpha_t_bar)) * (x- torch.sqrt(1-alpha_t_bar )*epsilon)
        x_hat_0=torch.clamp(x_hat_0,-1,1) # maye 0,1  # estimate original 

        # computing meant
        mean = ((torch.sqrt(alpha_t)*(1-alpha_t_bar_prev))/(1-alpha_t_bar))*x + ((torch.sqrt(alpha_t_bar_prev)*(1-alpha_t))/(1-alpha_t_bar)) * x_hat_0
       
        # Add noise unless it's the final step
        if t_index == 0:
            return mean # since mean + 0*x = mean
        else:
            noise = self.noise_like(x.shape, x.device)  # Generate noise z 
            sigma_t = torch.sqrt( ((1-alpha_t_bar_prev)/ (1-alpha_t_bar)) * (1-alpha_t) ) 
            return mean + sigma_t * noise # returns x t-1

        # ####################################################

    @torch.no_grad()
    def p_sample_loop(self, img):
        """
        Passes noise through the entire reverse diffusion process to generate
        final image samples.
        Args:
            img: The initial noise that is randomly sampled from the noise distribution.
        Returns:
            The sampled images.
        """
        b = img.shape[0]
        #### TODO: Implement the p_sample_loop function ####
        # 1. loop through the time steps from the last to the first
        # 2. inside the loop, sample x_{t-1} from the reverse diffusion process
        # 3. clamp and unnormalize the generated image to valid pixel range
        # Hint: to get time index, you can use torch.full()

        for i in reversed(range(self.num_timesteps)):
            t = torch.full((b,), i, dtype=torch.long, device=img.device)
            img = self.p_sample(img, t, i)
        img = torch.clamp(unnormalize_to_zero_to_one(img), 0, 1)

        return img
        # ####################################################

    @torch.no_grad()
    def sample(self, batch_size):
        """
        Wrapper function for p_sample_loop.
        Args:
            batch_size: The number of images to sample.
        Returns:
            The sampled images.
        """
        self.model.eval()
        #### TODO: Implement the sample function ####
        # Hint: use self.noise_like function to generate noise. DO NOT USE torch.randn
        img = self.noise_like((batch_size, self.channels, self.image_size, self.image_size), device=self.device)
        img = self.p_sample_loop(img)
        return img

    # forward diffusion
    def q_sample(self, x_0, t, noise):
        """
        Applies alpha interpolation between x_0 and noise to simulate sampling
        x_t from the noise distribution.
        Args:
            x_0: The initial images.
            t: 1D tensor containing a batch of time indices to sample at.
            noise: The noise tensor to sample from.
        Returns:
            The sampled images.
        """
        ###### TODO: Implement the q_sample function #######
        coef_mean = extract(self.sqrt_alphas_cumprod, t, x_0.shape).to(self.device)
        coef_var = extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape).to(self.device)
        return coef_mean * x_0 + coef_var * noise

    def p_losses(self, x_0, t, noise):
        """
        Computes the loss for the forward diffusion.
        Args:
            x_0: The initial images.
            t: 1D tensor containing a batch of time indices to compute the loss at.
            noise: The noise tensor to use.
        Returns:
            The computed loss.
        """
        ###### TODO: Implement the p_losses function #######
        # define loss function wrt. the model output and the target
        # Hint: you can use pytorch built-in loss functions: F.l1_loss
        
        x_t = self.q_sample(x_0, t, noise)
        pred_noise = self.model(x_t, t)
        loss = F.l1_loss(pred_noise, noise)

        return loss

        # ####################################################

    def forward(self, x_0, noise):
        """
        Acts as a wrapper for p_losses.
        Args:
            x_0: The initial images.
            noise: The noise tensor to use.
        Returns:
            The computed loss.
        """
        b, c, h, w, device, img_size, = *x_0.shape, x_0.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        ###### TODO: Implement the forward function #######
        
        t = torch.randint(0, self.num_timesteps, (b,), dtype=torch.long, device=device)
        return self.p_losses(x_0, t, noise)