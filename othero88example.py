from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable

# 計算を行うモデル
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(2, 64, 3, 1)
        self.fc1 = nn.Linear(576, 128)
        self.fc2 = nn.Linear(128, 64)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x)
        return output

# -1(敵)と1(味方)を分類するモデル
class BWClassify(nn.Module):
    def __init(self):
        super(BWClassify, self).__init__()
    def forward(self, x):
        x1 = torch.where(x == 1, torch.ones(8, 8), torch.zeros(8, 8))
        x2 = torch.where(x == -1, torch.ones(8, 8), torch.zeros(8, 8))
        x = torch.stack([x1, x2], dim=1)
        output = x
        return output

# 読み込んだモデルはこれを使用する。
class OutLayer(nn.Module):
    def __init__(self):
        super(OutLayer, self).__init__()
        self.bw = BWClassify()
        self.net = Net()
    def forward(self, x):
        x = self.bw(x)
        out = self.net(x)
        return out


# OutLayerのモデルを使いやすい値に変換する。
class MaxLayer(nn.Module):
    def __init__(self):
        super(MaxLayer, self).__init__()
    def forward(self, x):
        maxP = int(x.argmax())
        xpos = maxP // 8
        ypos = maxP % 8
        return {'x': xpos, 'y': ypos}

model = OutLayer()
model.load_state_dict(torch.load("othero_NN_88.pt"))
model.eval()

# 1は味方、-1は敵。ここまでは事前に処理すること。
board = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, -1, 1, 0, 0, 0],
    [0, 0, 0, 1, -1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]
]
# 盤面はテンソルに変換すること。
# n*8*8を受け取る。
bt = torch.tensor([board], dtype=torch.float32)

ans = model.forward(bt)
ansMax = ans.argmax()

# 読み込んだモデルの使用例
print(ansMax)
# xが縦向き、yが横向きなので注意。
print("x: " + str(ansMax//8) + ", y: " + str(ansMax%8))

# 読み込んだモデルをさらに層に通し、最大のインデックスをx, y座標に変換する
maxLayer = MaxLayer()
print(maxLayer(ans))