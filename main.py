import torch
from argparse import ArgumentParser
import os
from torch.backends import cudnn
from pathlib import Path
from train import train
from eval import test
from model import Model


def main(config):
    cudnn.beta_1 = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(config.save_weights_dir, exist_ok=True)
    model = Model(T=config.T, ch=config.channel, ch_mult=config.channel_mult,
                  attn=config.attn, num_res_blocks=config.num_res_blocks, dropout=config.dropout).to(device)
    print(model)
    if config.mode == 'train':
        train(model, config)
    else:
        test(model, config)


if __name__ == '__main__':
    parser = ArgumentParser()

    # parser.add_argument('--gpu', type=str, default="cuda:0", choices=["cuda:0", "cpu"])
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])

    # Train
    parser.add_argument('--dataset', type=str, default="Cifar-10", choices=["CelebA", 'Cifar-10'], )
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--num_epoch', type=int, default=200)
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--T', type=int, default=1000)
    parser.add_argument('--beta_1', type=float, default=1e-4)
    parser.add_argument('--beta_T', type=float, default=0.02)
    parser.add_argument('--grad_clip', type=int, default=1)
    parser.add_argument('--save_weights_dir', type=str, default="./checkpoints/")
    parser.add_argument('--test_frequency', type=int, default="10")

    #Loading file
    parser.add_argument('--training_load_weight', type=str, default=None)

    #Generation
    parser.add_argument('--sampled_dir', type=str, default="./sampledimgs/")
    parser.add_argument('--sampledNoisyImgName', type=str, default="NoisyNoGuidenceImgs.png")
    parser.add_argument('--sampledImgName', type=str, default="SampledNoGuidenceImgs")
    parser.add_argument('--write_frequency', type=int, default="10")
    parser.add_argument('--multiplier', type=float, default=2.)
    parser.add_argument('--nrow', type=int, default=8)


    #model parameters
    parser.add_argument('--channel', type=int, default=128)
    parser.add_argument('--channel_mult', type=int, default=[1, 2, 3, 4])
    parser.add_argument('--attn', type=int, default=[2])
    parser.add_argument('--num_res_blocks', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.15)

    config = parser.parse_args()
    print(config)
    main(config)
