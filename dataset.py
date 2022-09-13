from torchvision import transforms
from torchvision.datasets import CIFAR10, CelebA
from torch.utils.data import DataLoader


class CifarDataset():
    def __init__(self):
        self._cifar_dataset = CIFAR10(
            root='./data',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                # Correct normalization values for CIFAR-10: (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
            ]
            )
        )

    def Dataloader(self, config):
        dataloader = DataLoader(
            self._cifar_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True, )
        return dataloader

class CelebADataset():
    def __init__(self):
        self._celeba_dataset = CelebA(
            root='./data',
            split="train",
            download=True,
            transform=transforms.Compose([
                transforms.Resize((32,32)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))

    def Dataloader(self, config):
        dataloader = DataLoader(
            self._celeba_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True, )
        return dataloader