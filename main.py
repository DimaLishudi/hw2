import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid

from modeling.diffusion import DiffusionModel
from modeling.training import generate_samples, train_epoch
from modeling.unet import UnetModel

import os
import wandb
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig


# no bug fixes in main, only refactoring to handle hyperparameters via hydra
@hydra.main(version_base=None, config_path=".", config_name="params.yaml")
def main(cfg: DictConfig):
    cfg = cfg.train
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ddpm = DiffusionModel(
        eps_model=UnetModel(**cfg.model.unet),
        **cfg.model.ddpm
    )
    ddpm.to(device)

    # instantiate optimizer, scheduler and augmentations
    optim = instantiate(cfg.optimizer, params=ddpm.parameters())
    scheduler =  instantiate(cfg.scheduler, optimizer=optim)
    
    # Normalize could be added to cfg, but I decided to keep it hardcoded
    transform_list = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    if "augmentations" in cfg.dataloader:
        for aug in cfg.dataloader.augmentations:
            transform_list.append(hydra.utils.instantiate(aug))
    train_transforms = transforms.Compose(transform_list)

    dataset = CIFAR10(
        "cifar10",
        train=True,
        download=False,
        transform=train_transforms,
    )
    dataloader = DataLoader(dataset, **cfg.dataloader.params)

    # some initial logging
    enable_logging = cfg.logger.enable_logging
    if enable_logging:
        run = wandb.init(config=cfg, project=cfg.logger.project)
        params_path = os.getcwd() + "/params.yaml"
        run.log_code(name="config", include_fn=lambda s: s==params_path)
        sample_batch = make_grid(next(iter(dataloader))[0], nrow=16)
        wandb.log({"Dataset Samples": wandb.Image(sample_batch)}, commit=False)
    
    os.makedirs("samples", exist_ok=True)
    for i in range(cfg.training.num_epochs):
        if enable_logging:
            wandb.log({ "epoch" : i}, commit=False)
        train_epoch(ddpm, dataloader, optim, scheduler, device, enable_logging)
        generate_samples(ddpm, device, f"samples/{i:02d}.png", enable_logging)
    if enable_logging:
        wandb.finish()


if __name__ == "__main__":
    main()
