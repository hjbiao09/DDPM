from sched import scheduler

import torch
from dataset import CifarDataset
from model import Model
from tqdm.auto import tqdm
import torch.nn as nn
import torch.nn.functional as F
from utils import extract, Adder
import torch.optim as optim
from dataset import CifarDataset, CelebADataset
from pynvml import *
from torch.utils.tensorboard import SummaryWriter
from utils import GradualWarmupScheduler
from eval import test


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(model, config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    
    if config.dataset == 'Cifar-10':
        cifardatset = CifarDataset()
        dataloader = cifardatset.Dataloader(config)
        print("Training with Cifar-10")
    else:
        CelebAdatset = CelebADataset()
        dataloader = CelebAdatset.Dataloader(config)
        print(f"Training with CelebA...")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)

    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=config.num_epoch, eta_min=0, last_epoch=-1)

    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, multiplier=config.multiplier, total_epoch=config.num_epoch // 10,
        after_scheduler=cosineScheduler)
    '''
    Increase lr from base_lr to base_lr * multiplier in total_epoch, then change lr to after_scheduler
    '''

    if config.training_load_weight is not None:
        state = torch.load(config.training_load_weight)
        start_epoch = state["last_epoch"]
        optimizer.load_state_dict(state["optimizer"])
        warmUpScheduler.load_state_dict(state["scheduler"])
        model.load_state_dict(state["model"])
        print(f"{config.training_load_weight} has been loaded...")
    else:
        start_epoch = 0
        print("Start a new training...")

    trainer = GaussianDiffusionTrainer(model, config.beta_1, config.beta_T, config.T).to(device)
    os.makedirs("./logdir", exist_ok=True)
    writer = SummaryWriter(log_dir="./logdir/")
    iter_loss_adder = Adder()

    for epoch in range(start_epoch, config.num_epoch):

        losses = []
        loop = tqdm(enumerate(dataloader), total=len(dataloader))

        for batch_idx, (data, target) in loop:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = trainer(data)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.grad_clip)
            optimizer.step()

            if batch_idx == 0:
                nvmlInit()
                h = nvmlDeviceGetHandleByIndex(0)
                info = nvmlDeviceGetMemoryInfo(h)
            losses.append(loss.item())
            iter_loss_adder(loss.item())

            if batch_idx % config.write_frequency == 0:
                writer.add_scalar("Training loss", iter_loss_adder.average(),
                                  global_step=epoch * len(dataloader) + batch_idx)
                iter_loss_adder.reset()

            loop.set_description(f'Epoch [{epoch}/{config.num_epoch}] '
                                 f'GPU [{round(info.used / 1024 ** 3, 1)} GB/'
                                 f'{round(info.total / 1024 ** 3, 1)} GB]')
            loop.set_postfix(loss=loss.item(),
                             LR=optimizer.state_dict()['param_groups'][0]["lr"])

        warmUpScheduler.step()

        os.makedirs("./model/", exist_ok=True)
        torch.save({"model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": warmUpScheduler.state_dict(),
                    "last_epoch": epoch}, "./model/model_last.pkl")
        if epoch % config.test_frequency ==0:
            train_batch_size = config.batch_size
            test(model, config, epoch, eval=False)
            config.batch_size = train_batch_size
            model.train()


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()
        self.model = model
        self.T = T
        self.register_buffer(
            "betas", torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        # Returns the cumulative product of elements of input in the dimension dim.
        # e.g. y_i = x_1 * x_2 * x_3 * ... * x_i

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_bar", torch.sqrt(alphas_bar))
        self.register_buffer("sqrt_one_minus_alphas_bar", torch.sqrt(1. - alphas_bar))

    def forward(self, x_0):
        """
        There is no need to proceed with all forwards in the learning process.
        The q(x_{t-1} \mid x_t, x_0) is associated with the input image x_0 and \epsilon.
        Therefore, t can be sampled in a uniform manner and learned one by one.
        """

        t = torch.randint(self.T, size=(x_0.shape[0],), device=device)  # step t, uniform하게 학습.
        noise = torch.randn_like(x_0)
        # normal distribution 생성
        x_t = (
                extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
                extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        predict_noise = self.model(x_t, t)
        loss = F.mse_loss(predict_noise, noise)
        return loss
