import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import mlflow
import mlflow.pytorch

# ------------------------------
# Config
# ------------------------------
BATCH_SIZE = 64
EPOCHS = 3
LR = 0.001
MLFLOW_TRACKING_URI = "file:./mlruns"  # or your MLflow server
MODEL_NAME = "MNIST_CNN"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# ------------------------------
# Simple CNN
# ------------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32*7*7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(2,2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32*7*7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ------------------------------
# Data
# ------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform),
                          batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(datasets.MNIST('./data', train=False, download=True, transform=transform),
                         batch_size=BATCH_SIZE, shuffle=False)

# ------------------------------
# Train & Evaluate
# ------------------------------
def train(model, device, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

def evaluate(model, device, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            preds = output.argmax(dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)
    return correct / total

# ------------------------------
# Main Script
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# ------------------------------
# MLflow CT+CD Logic
# ------------------------------
def get_latest_production_accuracy(model_name):
    try:
        client = mlflow.tracking.MlflowClient()
        prod_versions = client.get_latest_versions(model_name, stages=["Production"])
        if len(prod_versions) == 0:
            return 0.0
        run_id = prod_versions[0].run_id
        run_info = client.get_run(run_id)
        return run_info.data.metrics.get("test_accuracy", 0.0)
    except Exception:
        return 0.0

baseline_acc = get_latest_production_accuracy(MODEL_NAME)
print(f"Latest Production Accuracy: {baseline_acc:.4f}")

with mlflow.start_run(run_name="MNIST_CNN_CTCD"):

    # Train loop
    for epoch in range(1, EPOCHS+1):
        loss = train(model, device, train_loader, optimizer, criterion)
        acc = evaluate(model, device, test_loader)
        print(f"Epoch {epoch}: Loss={loss:.4f}, Test Accuracy={acc:.4f}")

        # Log metrics
        mlflow.log_metric("train_loss", loss, step=epoch)
        mlflow.log_metric("test_accuracy", acc, step=epoch)

    # Compare & register only if improved
    if acc > baseline_acc:
        print(f"New model accuracy {acc:.4f} > baseline {baseline_acc:.4f} → registering in MLflow Model Registry")
        mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            registered_model_name=MODEL_NAME
        )
        # Promote to Production stage
        client = mlflow.tracking.MlflowClient()
        latest_version = client.get_latest_versions(MODEL_NAME)[-1].version
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=latest_version,
            stage="Production"
        )
        print(f"Model promoted to Production")
    else:
        print(f"New model did not improve. Skipping registration.")
