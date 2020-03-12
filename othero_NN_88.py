from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
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

class BWClassify(nn.Module):
    def __init(self):
        super(BWClassify, self).__init__()
    def forward(self, x):
        x1 = torch.where(x == 1, torch.ones(8, 8), torch.zeros(8, 8))
        x2 = torch.where(x == -1, torch.ones(8, 8), torch.zeros(8, 8))
        x = torch.stack([x1, x2], dim=1)
        output = x
        return output

class MaxLayer(nn.Module):
    def __init__(self):
        super(MaxLayer, self).__init__()
    def forward(self, x):
        maxP = x.argmax()
        xpos = maxP % 8
        ypos = maxP // 8
        return torch.tensor([xpos, ypos], dtype=torch.int)

class OutLayer(nn.Module):
    def __init__(self, model):
        super(OutLayer, self).__init__()
        self.model = model
    def forward(self, x):
        bwClassify = BWClassify()
        maxLayer = MaxLayer()
        x = self.model(bwClassify(x))
        return maxLayer(x)

def train(model: Net, device: torch.device, data: torch.tensor, target: torch.tensor, optimizer: optim.Adadelta, epoch: int):
    
    model.train()
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)

    loss = F.nll_loss(output, target) # outputはone-hotの形式、targetは正解値の形式で比較。
    loss.backward()
    optimizer.step()

def main():

    torch.manual_seed(1)

    device = torch.device("cpu")
    #device = torch.device("cuda")


    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=1.0)

    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

    DataPath = "kihuData2.txt"
    TargetPath = "kihuTargets2.txt"

    Datas = []
    with open(DataPath) as f:
        lineNum = 0
        boardData = [[0 for i in range(8)] for j in range (8)]
        for line in f:
            if lineNum == 0:
                boardData = [[0 for i in range(8)] for j in range (8)]
            l = line.split(', ')
            data = []
            for i in l:
                data.append(float(i.rstrip(('\n'))))
            boardData[lineNum] = data
            lineNum += 1
            if lineNum == 8:
                lineNum = 0
                Datas.append(boardData)
    for i in range(10):
        print(Datas[i])

    Targets = []
    with open(TargetPath) as f:
        for num in f:
            Targets.append(int(num.rstrip('\n')))

    datas = torch.tensor(Datas, dtype=torch.float32)
    targets = torch.tensor(Targets, dtype=torch.int64)
    datas = datas[: 100]
    targets = targets[: 100] # 10000件の学習。バッチ指定なしの状態では多すぎるとメモリ不足になる


    datas = datas.to(device)

    targets = targets.to(device)
    bwClassify = BWClassify()
    bwClassify = bwClassify.to(device)
    dataBW = bwClassify(datas)
    dataBW = dataBW.to(device)

    epochs = 10
    for epoch in range(1, epochs + 1):
        print("epoch: " + str(epoch))
        train(model, device, dataBW, targets, optimizer, epoch)
        scheduler.step()

    correct = 0
    trials = 140000
    for i in range(trials):
        testData = torch.tensor([Datas[i]], dtype=torch.float32)
        testData = testData.to(device)
        testdataBW = bwClassify(testData)
        testdataBW = testdataBW.to(device)
        ans = model.forward(testdataBW) # 現状のまま実用するなら、この値を上から順に合法手か探す。
        ansMax = ans.argmax(dim=1, keepdim=True)
        Tar = Targets[i]
        if(ansMax == Tar):
            correct += 1
    print('correct: ' + str(correct) + ' / ' + str(trials))
    print('Rating' + str(correct / trials))
    outLayer = OutLayer(model)
    outLayer.eval()
    trials = 10
    for i in range(trials):
        testData = torch.tensor([Datas[i]], dtype=torch.float32)
        testData = testData.to(device)
        ans = outLayer.forward(testData) # 現状のまま実用するなら、この値を上から順に合法手か探す。
        print(Datas[i])
        print(ans)
    torch.save(outLayer.state_dict(), "othero_NN_88.pt")

    # x = index % 8
    # y = int(index / 8)
    #でインデックスとして取ってるデータを座標に変換可能。


if __name__ == '__main__':
    main()