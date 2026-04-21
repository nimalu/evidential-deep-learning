import torch
import numpy as np
from torch import nn
from tqdm import tqdm
import warnings
from edl import build_cifar_loaders, EDLModel, edl_loss, validate_model

warnings.filterwarnings("ignore", category=DeprecationWarning)

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


EPOCHS = 50
LEARNING_RATE = 1e-3
OUTPUT_PATH = "edl_cifar10_resnet20.pth"


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    learning_rate: float = LEARNING_RATE,
    epochs: int = EPOCHS,
    T=20,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    warmup_epochs = int(0.05 * epochs)  # 5% of total epochs for warmup

    # 1. Warmup phase: linearly increase LR from 1% of base LR to 100% over warmup_epochs
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=warmup_epochs
    )

    # 2. Main phase: Decays LR by 0.1 at 50%, 75% of training
    main_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(epochs * 0.5), int(epochs * 0.75)], gamma=0.1
    )

    # 3. Combine them: Run warmup first, then switch to main scheduler at milestone
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_epochs],
    )

    model.to(device)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs}",
            unit="batch",
            ncols=100,
            mininterval=0.5,
        )

        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            evidence = model(inputs)
            loss = edl_loss(evidence, labels, epoch, num_classes=10, T=T, gamma=1.0)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            running_loss += loss.item()

            progress_bar.set_postfix({"loss": running_loss / (progress_bar.n + 1)})

        scheduler.step()

        model.eval()
        accuracy, total_loss = validate_model(model, test_loader)
        torch.save(model.state_dict(), OUTPUT_PATH)

        print(f"Validation Accuracy: {accuracy * 100:.2f}%")
        print(f"Validation Loss: {total_loss:.4f}")


if __name__ == "__main__":
    model = EDLModel(pretrained=True)
    loaders = build_cifar_loaders(train_batch_size=256, test_batch_size=256)
    train_loader = loaders["cifar10_loader_train"]
    test_loader = loaders["cifar10_loader_test"]

    train_model(model, train_loader, test_loader)
