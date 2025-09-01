import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#定义模型
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x
    #尽量把数据放在一个类中定义，这样有很多方法也可以随着类定义下来容易调用
class Data(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + self.seq_length]
        x = x.reshape(-1, 1)#这里的reshape是为了符合LSTM的输入要求(batch, seq_len, input_size)
        return torch.tensor(x, dtype=torch.float32), torch.tensor([y], dtype=torch.float32)

array = np.array(np.sin(np.arange(0, 100, 0.1)))
print(array.shape)

train_size= 80
seq_length = 10
trainD = array[:train_size]
testD = array[train_size:]
train_Set = Data(trainD, seq_length)
test_Set = Data(testD, seq_length)

train_loader = DataLoader(train_Set, batch_size=16, shuffle=True)
test_loader = DataLoader(test_Set, batch_size=16, shuffle=False)

input_size = 1
hidden_size = 64
output_size = 1
model = LSTM(input_size, hidden_size, output_size).to(device)
print(model)

round = 0
epoch = 50
learning_rate = 0.001

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), 
                      lr=learning_rate)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                mode='max', 
                                                factor=0.5, 
                                                patience=3)

for i in range(epoch):
     round += 1
     model.train()
     train_loss = 0.0

     for x, y in train_loader:
         x, y = x.to(device), y.to(device)
         optimizer.zero_grad()
         outputs = model(x)
         loss = criterion(outputs, y)
         loss.backward()
         optimizer.step()
         train_loss += loss.item() * x.size(0)

     print(f"Epoch [{round}/{epoch}], Loss: {train_loss/len(train_loader):.4f}")

model.eval()
perdiction = []
real = []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        outputs = model(x)
        perdiction.append(outputs.cpu().numpy())
        real.append(y.cpu().numpy())

    perdiction = np.concatenate(perdiction).reshape(-1)
    real = np.concatenate(real).reshape(-1)

plt.figure(figsize=(12, 6))
plt.plot(perdiction, label='Predicted')
plt.plot(real, label='Actual')
plt.legend()
plt.show()

torch.save(model.state_dict(), 'lstm_model.pth')