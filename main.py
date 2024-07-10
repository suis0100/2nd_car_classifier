import numpy as np
import torch
import pandas as pd
from urllib.request import urlretrieve
import torch.utils.data as Data
import torch.nn as nn
import matplotlib.pyplot as plt


# 从文件中加载数据
def load_data(download=False):      # 需要下数据就改成True
    if download:
        data_path, _ = urlretrieve("http://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data", "car.csv")
        print("Downloaded to car.csv")

    col_names = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
    data = pd.read_csv("D:/py_code/SD/car+evaluation/car.data", names=col_names)
    return data
# car_data = load_data()
# print(car_data)
# 每列数据类型
# for i in range(car_data.columns.size):
#     temp_list = car_data.iloc[:, i].str.split(':')
#     cate_list = list(set([i[0] for i in temp_list]))
#     print(car_data.columns[i], cate_list)


def convert2onehot(data):
    return pd.get_dummies(data, prefix=data.columns)
# car_data = load_data()
# car_data_onehot = convert2onehot(car_data)
# print(car_data_onehot)


# 网络
class CLASSIFIER(nn.Module):
    def __init__(self):
        super(CLASSIFIER, self).__init__()

        self.L1 = nn.Sequential(
            nn.Linear(21, 128),
            nn.ReLU()
        )
        self.L2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.out = nn.Linear(128, 4)

    def forward(self, x):
        x = self.L1(x)
        x = self.L2(x)
        output = self.out(x)
        return output


# parameters
np.random.seed(1)
BATCH_SIZE = 32
LR = 0.001
EPOCH = 10

data = load_data()
new_data = convert2onehot(data)
new_data = new_data.to_numpy(np.float32)
# print(new_data)
np.random.shuffle(new_data)
sep = int(0.7*len(new_data))
train_data = new_data[:sep, :]
test_data = new_data[sep:, :]
# 训练数据集和测试数据
train_x = torch.from_numpy(train_data[:, :21]).cuda().float()
train_y = torch.from_numpy(train_data[:, 21:]).cuda().float()
train_dataset = Data.TensorDataset(train_x, train_y)
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_x = torch.from_numpy(test_data[:, :21]).cuda().float()
test_y = torch.from_numpy(test_data[:, 21:]).float()

classifier = CLASSIFIER().cuda()
# print(classifier)

optimizer = torch.optim.Adam(classifier.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):
        output = classifier(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_output = classifier(test_x)
            pred_y = torch.max(test_output, 1)[1].data.cpu().numpy()
            test_y_new = torch.max(test_y, 1)[1].data.numpy()
            accuracy = float((pred_y == test_y_new).astype(int).sum()) / float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy(), '| test accuracy: %.2f' % accuracy)




test_output = classifier(test_x[100:110])
pred_y = torch.max(test_output, 1)[1].data.cpu().numpy()
print(pred_y, 'prediction number')
print(test_y_new[100:110], 'real number')



