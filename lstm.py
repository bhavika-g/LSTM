import torch
import torch.nn as nn
import torch.nn.functional as F

class SeriesPredictor(nn.Module):
    def __init__(self, input_length, dropout=0.3):
        super(SeriesPredictor, self).__init__()
        
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=10, kernel_size=300, padding=149)
        self.act1 = nn.SiLU() 
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(in_channels=10, out_channels=10, kernel_size=10, padding=5)
        self.act2 = nn.ELU()
        self.drop2 = nn.Dropout(dropout)

        self.conv3 = nn.Conv1d(in_channels=10, out_channels=10, kernel_size=5, padding=2)
        self.act3 = nn.ELU()
        self.drop3 = nn.Dropout(dropout)

 
        self.lstm = nn.LSTM(input_size=10, hidden_size=32, num_layers=10,
                            batch_first=True, bidirectional=True)

  
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=8, kernel_size=5, padding=2)
        self.conv5 = nn.Conv1d(in_channels=8, out_channels=4, kernel_size=5, padding=2)
        self.conv6 = nn.Conv1d(in_channels=4, out_channels=2, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=5, padding=2)

    def forward(self, x):  
        x = x.unsqueeze(1)  
        x = self.drop1(self.act1(self.conv1(x)))
        x = self.drop2(self.act2(self.conv2(x)))
        x = self.drop3(self.act3(self.conv3(x)))      
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)        
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = x.permute(0, 2, 1) 
        return x

class CustomSmoothMSELoss(nn.Module):
    def __init__(self, smooth_weight=0.1):
        super().__init__()
        self.mse = nn.MSELoss()
        self.smooth_weight = smooth_weight

    def forward(self, pred, target):
        # MSE term
        loss = self.mse(pred, target)

        # Smoothness penalty: L2 norm of difference between time steps
        diff = pred[:, 1:, :] - pred[:, :-1, :]
        smoothness = torch.mean(diff ** 2)

        return loss + self.smooth_weight * smoothness

class DummyCalciumDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples=100, seq_len=1000):
        self.data = torch.randn(num_samples, seq_len)
        self.labels = torch.sin(self.data * 0.5)  

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device).unsqueeze(-1)

        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device).unsqueeze(-1)
            output = model(x)
            loss = criterion(output, y)
            total_loss += loss.item()
    return total_loss / len(loader)


from torch.utils.data import DataLoader
import torch.optim as optim

# Hyperparameters
EPOCHS = 10
BATCH_SIZE = 8
LR = 1e-3

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
model = SeriesPredictor(input_length=1000).to(device)

# Data
train_data = DummyCalciumDataset(num_samples=80)
test_data = DummyCalciumDataset(num_samples=20)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

# Loss & Optimizer
criterion = CustomSmoothMSELoss(smooth_weight=0.05)
optimizer = optim.Adam(model.parameters(), lr=LR)

# Training Loop
for epoch in range(EPOCHS):
    train_loss = train(model, train_loader, optimizer, criterion, device)
    test_loss = evaluate(model, test_loader, criterion, device)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
