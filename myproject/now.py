import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

# 定义分类模型
class ClassificationModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ClassificationModel, self).__init__()

        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, num_classes)
        self.dropout = nn.Dropout(0.1)

        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x


# 定义回归模型
class RegressionModel0(nn.Module):
    def __init__(self):
        super(RegressionModel0, self).__init__()

        self.fc1 = nn.Linear(61,128)
        self.fc2 = nn.Linear(128,32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)
        return x
class RegressionModel1(nn.Module):
    def __init__(self):
        super(RegressionModel1, self).__init__()

        self.fc1 = nn.Linear(61, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 16)
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
class RegressionModel2(nn.Module):
    def __init__(self):
        super(RegressionModel2, self).__init__()

        self.fc1 = nn.Linear(61, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 16)
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
class RegressionModel3(nn.Module):
    def __init__(self):
        super(RegressionModel3, self).__init__()

        self.fc1 = nn.Linear(61,128)
        self.fc2 = nn.Linear(128,32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
# 定义自定义数据集
class MyDataset(Dataset):
    def __init__(self, features, target_class, target_reg):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.target_class = torch.tensor(target_class, dtype=torch.long)
        self.target_reg = torch.tensor(target_reg, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        x = self.features[index]
        y_class = self.target_class[index]
        y_reg = self.target_reg[index]
        return x, y_class, y_reg


# 加载数据
file_path = r'C:\Users\Administrator\Desktop\now1.xlsx'
raw_data = pd.read_excel(file_path, header=0)

features = raw_data.iloc[:, :-2].values
target_class = raw_data.iloc[:, -1].values
target_reg = raw_data.iloc[:, -2].values

# 数据预处理
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
batch_size = 256

# 划分训练集和测试集
train_data, test_data, train_target_class, test_target_class, train_target_reg, test_target_reg = train_test_split(
    scaled_features, target_class, target_reg, test_size=0.2, random_state=42
)

# 创建训练集和测试集的数据集和数据加载器
train_dataset = MyDataset(train_data, train_target_class, train_target_reg)
test_dataset = MyDataset(test_data, test_target_class, test_target_reg)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 创建四个子训练集
train_dataset_0 = MyDataset(train_data[train_target_class == 0], train_target_class[train_target_class == 0],train_target_reg[train_target_class == 0])
train_dataset_1 = MyDataset(train_data[train_target_class == 1], train_target_class[train_target_class == 1],train_target_reg[train_target_class == 1])
train_dataset_2 = MyDataset(train_data[train_target_class == 2], train_target_class[train_target_class == 2],train_target_reg[train_target_class == 2])
train_dataset_3 = MyDataset(train_data[train_target_class == 3], train_target_class[train_target_class == 3], train_target_reg[train_target_class == 3])


# 创建四个子训练集的数据加载器
train_dataloader_0 = DataLoader(train_dataset_0, batch_size=batch_size, shuffle=True)
train_dataloader_1 = DataLoader(train_dataset_1, batch_size=batch_size, shuffle=True)
train_dataloader_2 = DataLoader(train_dataset_2, batch_size=batch_size, shuffle=True)
train_dataloader_3 = DataLoader(train_dataset_3, batch_size=batch_size, shuffle=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 创建模型实例
input_size = train_data.shape[1]

num_classes = 4
classification_model = ClassificationModel(input_size, num_classes).to(device)
regression_model_0 = RegressionModel0().to(device)
regression_model_1 = RegressionModel1().to(device)
regression_model_2 = RegressionModel2().to(device)
regression_model_3 = RegressionModel3().to(device)

# 定义损失函数和优化器
criterion_classification = nn.CrossEntropyLoss()
criterion_regression = nn.MSELoss()
optimizer_classification = optim.Adam(classification_model.parameters(), lr=0.00008,weight_decay=0.002)
optimizer_regression_0 = optim.Adam(regression_model_0.parameters(), lr=0.008, weight_decay=0.05)
optimizer_regression_1 = optim.Adam(regression_model_1.parameters(), lr=0.004, weight_decay=0.05)
optimizer_regression_2 = optim.Adam(regression_model_2.parameters(), lr=0.004, weight_decay=0.05)
optimizer_regression_3 = optim.Adam(regression_model_3.parameters(), lr=0.005)

# 定义分类训练函数


def train_classification(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    loss_values = []  # 用于保存每个epoch的平均损失值
    accuracy_values = []  # 用于保存每个epoch的准确率值


    for inputs, targets_class, _ in dataloader:
        inputs = inputs.to(device)
        targets_class = targets_class.to(device)

        optimizer.zero_grad()
        outputs_class = model(inputs)
        loss = criterion(outputs_class, targets_class)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        _, predicted_class = torch.max(outputs_class.data, 1)
        total += targets_class.size(0)
        correct += (predicted_class == targets_class).sum().item()

    epoch_loss = running_loss / len(dataloader)
    accuracy = correct / total

    loss_values.append(epoch_loss)
    accuracy_values.append(accuracy)


    return epoch_loss, accuracy
# 定义分类测试函数
def test_classification(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets_class, _ in dataloader:
            inputs = inputs.to(device)
            targets_class = targets_class.to(device)

            outputs_class = model(inputs)

            loss = criterion(outputs_class, targets_class)

            running_loss += loss.item()
            _, predicted_class = torch.max(outputs_class.data, 1)
            total += targets_class.size(0)
            correct += (predicted_class == targets_class).sum().item()

    epoch_loss = running_loss / len(dataloader)
    accuracy = correct / total

    return epoch_loss, accuracy


# 定义回归训练函数
def train_regression(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0

    for inputs, _, targets_reg in dataloader:
        inputs = inputs.to(device)
        targets_reg = targets_reg.to(device)

        optimizer.zero_grad()
        outputs_reg = model(inputs)
        # 计算相对误差
        relative_errors = torch.abs((outputs_reg.squeeze() - targets_reg) / targets_reg)

        # 计算相对误差损失
        loss = criterion(outputs_reg.squeeze(), targets_reg) + torch.mean(relative_errors)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)

    return epoch_loss


# 定义回归测试函数


# 训练分类模型
num_epochs = 10000
loss_values_train = []  # 用于保存每个epoch的训练损失值
accuracy_values_train = []  # 用于保存每个epoch的训练准确率值
loss_values_test = []  # 用于保存每个epoch的测试损失值
accuracy_values_test = []  # 用于保存每个epoch的测试准确率值
for epoch in range(num_epochs):
    train_loss, train_accuracy = train_classification(classification_model, train_dataloader, criterion_classification, optimizer_classification)
    test_loss, test_accuracy = test_classification(classification_model, test_dataloader, criterion_classification)

    loss_values_train.append(train_loss)
    accuracy_values_train.append(train_accuracy)
    loss_values_test.append(test_loss)
    accuracy_values_test.append(test_accuracy)
    print(f"Epoch {epoch + 1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.4f}")
    print()

epochs = range(1, num_epochs + 1)

plt.figure(figsize=(10, 5))
plt.plot(epochs, loss_values_train, label='Train Loss')
plt.plot(epochs, loss_values_test, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Testing Loss')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(epochs, accuracy_values_train, label='Train Accuracy')
plt.plot(epochs, accuracy_values_test, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Testing Accuracy')
plt.legend()
plt.show()
# 根据分类结果训练回归模型
num_epochs_reg = 1

for epoch in range(num_epochs_reg):
    train_loss_0 = train_regression(regression_model_0, train_dataloader_0, criterion_regression, optimizer_regression_0)
    train_loss_1 = train_regression(regression_model_1, train_dataloader_1, criterion_regression, optimizer_regression_1)
    train_loss_2 = train_regression(regression_model_2, train_dataloader_2, criterion_regression, optimizer_regression_2)
    train_loss_3 = train_regression(regression_model_3, train_dataloader_3, criterion_regression, optimizer_regression_3)


    print(f"Epoch {epoch + 1}/{num_epochs_reg}")
    print(f"Train Loss (Class 0): {train_loss_0:.4f}")
    print(f"Train Loss (Class 1): {train_loss_1:.4f}")
    print(f"Train Loss (Class 2): {train_loss_2:.4f}")
    print(f"Train Loss (Class 3): {train_loss_3:.4f}")


# 测试集分类预测和回归预测
classification_model.eval()
regression_model_0.eval()
regression_model_1.eval()
regression_model_2.eval()
regression_model_3.eval()

predictions_class = []
predictions_reg = []

with torch.no_grad():
    for inputs, test_cls, test_res in test_dataloader:
        inputs = inputs.to(device)
        test_res = test_res.to(device)
        outputs_class = classification_model(inputs)
        _, predicted_class = torch.max(outputs_class.data, 1)
        #predicted_class 张量
        predictions_class.extend(predicted_class.tolist())
        #predictions_class 数组


        for i in range(len(predicted_class)):
            if predicted_class[i] == 0:
                outputs_reg = regression_model_0(inputs[i].unsqueeze(0))

            elif predicted_class[i] == 1:
                outputs_reg = regression_model_1(inputs[i].unsqueeze(0))

            elif predicted_class[i] == 2:
                outputs_reg = regression_model_2(inputs[i].unsqueeze(0))

            else:
                outputs_reg = regression_model_3(inputs[i].unsqueeze(0))

            # predictions_reg.append(outputs_reg.item())
            # print("Regression Predictions:", predictions_reg)


            print(f'Predicted: {outputs_reg.item():.2f}, True: {test_res[i]:.2f}')



# 打印分类预测和回归预测结果
print("分类预测值:", predictions_class)
print("分类真实值:", test_cls.tolist())

total_count = len(predictions_class)
correct_count = 0
for pred, true in zip(predictions_class, test_cls):
    if pred == true:
        correct_count += 1
accuracy = correct_count / total_count
print("分类准确率:", accuracy)
