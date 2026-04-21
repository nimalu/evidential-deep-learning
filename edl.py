import torch
from util import CIFAR_10_STATS
from torch import nn
from torch.nn import functional as F
import warnings
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms as T
import os

warnings.filterwarnings("ignore", category=DeprecationWarning)


class AddGaussianNoise(object):
    """Add Gaussian noise to a tensor image with given probability."""
    def __init__(self, mean=0.0, std=0.05, p=0.5):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, tensor):
        if torch.rand(1).item() < self.p:
            return tensor + torch.randn_like(tensor) * self.std + self.mean
        return tensor



def build_cifar_loaders(train_batch_size=64, test_batch_size=64):
    cifar10_transform_train = T.Compose(
        [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            AddGaussianNoise(mean=0.0, std=0.05, p=0.5),
            T.Normalize(CIFAR_10_STATS[0], CIFAR_10_STATS[1]),
        ]
    )
    cifar10_transform_test = T.Compose(
        [T.ToTensor(), T.Normalize(CIFAR_10_STATS[0], CIFAR_10_STATS[1])]
    )
    cifar10_train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=cifar10_transform_train
    )
    cifar10_test_set = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=cifar10_transform_test
    )
    cifar10_loader_train = DataLoader(
        cifar10_train_set,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=min((os.cpu_count() or 2) - 1, 4),
    )
    cifar10_loader_test = DataLoader(
        cifar10_test_set,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=min((os.cpu_count() or 2) - 1, 4),
    )
    cifar10_classes = cifar10_train_set.classes

    cifar100_test_set = torchvision.datasets.CIFAR100(
        root="./data", train=False, download=True, transform=cifar10_transform_test
    )
    cifar100_loader_test = DataLoader(
        cifar100_test_set, batch_size=test_batch_size, shuffle=False, num_workers=2
    )
    cifar100_classes = cifar100_test_set.classes
    return {
        "cifar10_loader_train": cifar10_loader_train,
        "cifar10_loader_test": cifar10_loader_test,
        "cifar10_classes": cifar10_classes,
        "cifar100_loader_test": cifar100_loader_test,
        "cifar100_classes": cifar100_classes,
    }


class EDLModel(nn.Module):
    def __init__(
        self,
        pretrained=False,
    ):
        super().__init__()
        self.model: nn.Module = torch.hub.load(
            "chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=pretrained
        )  # type: ignore
        if pretrained:
            # reset final layer to output evidence instead of logits
            self.model.fc = nn.Linear(self.model.fc.in_features, 10)

    def forward(self, x):
        return F.softplus(self.model(x))


def mse_dirichlet(alpha: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    S = torch.sum(alpha, dim=1, keepdim=True)
    p = alpha / S
    mse_term = torch.sum((y - p) ** 2, dim=1)
    variance_term = torch.sum((p * (1 - p)) / (S + 1), dim=1)
    return torch.mean(mse_term + variance_term)


def kl_dirichlet(alpha: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    alpha_tilde = y + (1 - y) * alpha
    beta = torch.ones_like(alpha_tilde)

    sum_alpha = torch.sum(alpha_tilde, dim=1)
    sum_beta = torch.sum(beta, dim=1)

    t1 = torch.lgamma(sum_alpha) - torch.lgamma(sum_beta)
    t2 = torch.sum(torch.lgamma(alpha_tilde), dim=1)
    t3 = alpha_tilde - beta
    t4 = torch.digamma(alpha_tilde) - torch.digamma(sum_alpha).unsqueeze(1)

    kl = t1 - t2 + torch.sum(t3 * t4, dim=1)
    return torch.mean(kl)


def edl_loss(
    evidence: torch.Tensor,
    labels: torch.Tensor,
    epoch: int,
    num_classes: int,
    T=10,
    gamma=1.0,
):
    y = F.one_hot(labels, num_classes=num_classes).float()
    alpha = evidence + 1
    mse = mse_dirichlet(alpha, y)
    kl = kl_dirichlet(alpha, y)
    annealing_coef = min(1.0, epoch / T)

    return mse + gamma * annealing_coef * kl


def validate_model(model: nn.Module, test_loader: DataLoader):
    device = next(model.parameters()).device

    model.eval()
    correct = total = 0
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            evidence = model(inputs)
            loss = edl_loss(evidence, labels, epoch=0, num_classes=10)
            total_loss += loss.item()
            _, predicted = torch.max(evidence, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total

    return accuracy, total_loss / len(test_loader)