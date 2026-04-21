import torch
import numpy as np
from tqdm import tqdm
import warnings
from edl import (
    build_cifar_loaders,
    EDLModel,
    edl_loss,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


EPOCHS = 200
LEARNING_RATE = 5e-3


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = EDLModel(pretrained=False)
loaders = build_cifar_loaders()
train_loader = loaders["cifar10_loader_train"]
test_loader = loaders["cifar10_loader_test"]


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)

warmup_epochs = int(0.05 * EPOCHS)  # 5% of total epochs for warmup

# 1. Warmup phase: linearly increase LR from 1% of base LR to 100% over warmup_epochs
warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.01, total_iters=warmup_epochs
)

# 2. Main phase: Decays LR by 0.1 at 50%, 75% of training
main_scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[int(EPOCHS * 0.5), int(EPOCHS * 0.75)], gamma=0.1
)

# 3. Combine them: Run warmup first, then switch to main scheduler at milestone
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs]
)

model.to(device)
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(
        train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", unit="batch", ncols=100
    )

    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        evidence = model(inputs)
        loss = edl_loss(evidence, labels, epoch, num_classes=10, T=100, gamma=1.0)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        running_loss += loss.item()

        progress_bar.set_postfix({"loss": running_loss / (progress_bar.n + 1)})

    scheduler.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            _, predicted = torch.max(model(inputs), dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Validation Accuracy: {100 * correct / total:.2f}%")


torch.save(model.state_dict(), "edl_cifar10_resnet20.pth")
