import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from modeling.diffusion import DiffusionModel
from modeling.training import generate_samples, train_epoch
from modeling.unet import UnetModel

import os
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

# no bug fixes in main, only refactoring to handle hyperparameters via hydra
@hydra.main(version_base=None, config_path="configs", config_name="base_config")
def main(config: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ddpm = DiffusionModel(
        eps_model=UnetModel(3, 3, hidden_size=128),
        betas=(1e-4, 0.02),
        num_timesteps=1000,
    )
    print(config)
    return
    ddpm.to(device)
    # Normalize could be added to config, but I decided to keep it hardcoded
    transform_list = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    for transform in config.data.augmentations:
        pass

    train_transforms = transforms.Compose(transform_list)

    dataset = CIFAR10(
        "cifar10",
        train=True,
        download=True,
        transform=train_transforms,
    )

    dataloader = DataLoader(dataset, batch_size=128, num_workers=4, shuffle=True)

    if enable_logging:
        run = wandb.init(config=config, project=config.Project)
        wandb.log_code("./configs/") 
        wandb.log({"Dataset Samples": next(iter(dataloader))}, commit=False)
    
    optim = torch.optim.Adam(ddpm.parameters(), lr=1e-5)

    os.makedirs("samples", exist_ok=True)

    for i in range(num_epochs):
        train_epoch(ddpm, dataloader, optim, device)
        generate_samples(ddpm, device, f"samples/{i:02d}.png")

    if enable_logging:
        wandb.finish()
    
if __name__ == "__main__":
    main()
