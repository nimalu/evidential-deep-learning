import torch
import torch.nn.functional as F
from tqdm import tqdm


def train(
    model,
    train_loader,
    test_loader,
    epochs,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    optimizer = model.configure_optimizer(num_samples=len(train_loader), epochs=epochs)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch"
        )

        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            loss = model.training_step((inputs, labels), epoch=epoch, epochs=epochs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            progress_bar.set_postfix(
                {"loss": running_loss / (progress_bar.n + 1)}
            )

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                _, predicted = torch.max(model(inputs), dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Validation Accuracy: {100 * correct / total:.2f}%")


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
    t2 = torch.sum(torch.lgamma(alpha_tilde) - torch.lgamma(beta), dim=1)
    t3 = alpha_tilde - beta
    t4 = torch.digamma(alpha_tilde) - torch.digamma(sum_alpha).unsqueeze(1)

    kl = t1 - t2 + torch.sum(t3 * t4, dim=1)
    return torch.mean(kl)


def edl_loss(
    alpha: torch.Tensor,
    labels: torch.Tensor,
    epoch: int,
    num_classes: int,
    T=10,
    gamma=0.1,
):
    y = F.one_hot(labels, num_classes=num_classes).float()

    mse = mse_dirichlet(alpha, y)
    kl = kl_dirichlet(alpha, y)
    annealing_coef = min(1.0, epoch / T)

    return mse + gamma * annealing_coef * kl
