from torchvision.datasets import CIFAR10

if __name__ == "__main__":
    train_dataset = CIFAR10("cifar10", train=True, download=True)