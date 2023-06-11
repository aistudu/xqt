import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import pandas as pd

from sklearn.preprocessing import StandardScaler,MinMaxScaler
import matplotlib.pyplot as plt

class MyDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32).to(device)
        self.targets = torch.tensor(targets, dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        x = self.features[index]
        y = self.targets[index]
        return x, y

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(61, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 4)
        self.fc5 = nn.Linear(4, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        return x
        #使用61，128-128，32-32-1时lr=0.004,e=2w过拟合


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batchsize = 64
file_path = r'C:\Users\Administrator\Desktop\1.xlsx'
raw_data = pd.read_excel(file_path, header=0)
features = raw_data.iloc[:, :-1].values
target = raw_data.iloc[:, -1].values
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
# print(scaled_features)
# normalizer = MinMaxScaler()
# normalized_features = normalizer.fit_transform(scaled_features)
ori_dataset=MyDataset(scaled_features,target)
train_data, test_data,train_target,test_target = train_test_split(scaled_features,target,test_size=0.2, random_state=42)
train_dataset = MyDataset(train_data, train_target)
test_dataset = MyDataset(test_data, test_target)
# print(train_dataset[:])
train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False)

model = Net().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.008,weight_decay=0.05)
#0
# 0.008,0.05 rmse 4.8 tl23.8,loss0.7
#1 目前预测的很好，但是为什么rmse这么高
mse_values = []
rmse_values = []
def train(epoch):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_dataloader,1):

        outputs = model(inputs)
        # 计算相对误差
        relative_errors = torch.abs((outputs.squeeze() - labels) / labels)

        # 计算相对误差损失
        loss = criterion(outputs.squeeze(), labels) + torch.mean(relative_errors)

        # loss = criterion(outputs.squeeze(), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if i % (len(train_dataset)/batchsize) == 0:
            print('[%d] loss: %.3f' % (epoch + 1, running_loss /(len(train_dataset)/batchsize)))
            running_loss = 0.0



def test(epoch):
    model.eval()
    test_loss = 0.0
    predictions = []
    targets = []
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            outputs = model(inputs)

            # relative_errors = torch.abs((outputs.squeeze() - labels) / labels)
            #
            # # 计算相对误差损失
            # loss = criterion(outputs.squeeze(), labels) + torch.mean(relative_errors)

            loss = criterion(outputs.squeeze(), labels)


            test_loss += loss.item()
            predictions.extend(outputs.squeeze().tolist())
            targets.extend(labels.tolist())

        mse = nn.MSELoss()
        rmse = torch.sqrt(mse(torch.tensor(predictions), torch.tensor(targets)))
        mse_values.append(test_loss / len(test_dataloader))
        rmse_values.append(rmse.item())
        print('Test RMSE: %.3f' % rmse.item())

        print('[%d] test_loss: %.3f' % (epoch + 1, test_loss / len(test_dataloader)))



if __name__ == '__main__':
    num_epochs = 3000


    for epoch in range(num_epochs):
        train(epoch)
        test(epoch)

    # model_name="loss{},test_loss{}".format(train().running_loss,test().loss)
    # print(model_name)
    # for i in train_dataset:
    #     # x=random.randint(0,len(test_dataset)-1)
    #     # print(i[0])
    #     # print(i[1])
    #     outputs = model(i[0].unsqueeze(0))
    #     print(f'Predicted: {outputs.item():.2f}, True: {i[1].item():.2f}')
    for i in test_dataset:
        # x=random.randint(0,len(test_dataset)-1)
        # print(i[0])
        # print(i[1])
        outputs = model(i[0].unsqueeze(0))
        print(f'Predicted: {outputs.item():.2f}, True: {i[1].item():.2f}')
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), mse_values, label='MSE')
    plt.plot(range(1, num_epochs + 1), rmse_values, label='RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()