import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import extract
from tqdm.auto import tqdm
import os
from torchvision.utils import save_image

class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, self.T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))

        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
                extract(self.coeff1, t, x_t.shape) * x_t -
                extract(self.coeff2, t, x_t.shape) * eps
        )

    def p_mean_variance(self, x_t, t):
        # below: only log_variance is used in the KL computations
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)

        eps_theta = self.model(x_t, t)
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps_theta)

        return xt_prev_mean, var

    def forward(self, x_T):

        x_t = x_T
        for step in tqdm((range(self.T))):
            time_step = 1000 - 1 - step
            # print(time_step)
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, var = self.p_mean_variance(x_t=x_t, t=t)
            # reparameter trick
            # x_{t-1} = \mu + \sigma * \epsilon(noise)
            # no noise when t == 0
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.sqrt(var) * noise
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t
        return torch.clip(x_0, -1, 1)

def test(model, config, epoch=200, eval=True):
    config.batch_size = 64
    with torch.no_grad():
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if eval:
            state_model = torch.load(config.training_load_weight)
            model.load_state_dict(state_model)
        model.eval()
        sampler = GaussianDiffusionSampler(
            model, config.beta_1, config.beta_T, config.T).to(device)
        # Sampled from standard normal distribution
        noisyImage = torch.randn(
            size=[config.batch_size, 3, 32, 32], device=device)
        saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
        os.makedirs(config.sampled_dir, exist_ok=True)
        save_image(saveNoisy, os.path.join(
            config.sampled_dir, config.sampledNoisyImgName), nrow=config.nrow)
        sampledImgs = sampler(noisyImage)
        sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
        save_image(sampledImgs, os.path.join(
            config.sampled_dir, f"{config.sampledImgName}_{epoch:03d}.png"), nrow=config.nrow)