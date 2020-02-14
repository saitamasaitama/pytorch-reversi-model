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
        self.fc1 = nn.Linear(128, 9216)
        self.fc2 = nn.Linear(9216, 64)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def train(args: str, model: Net, device: torch.device, data: torch.tensor, target: torch.tensor, optimizer: optim.Adadelta, epoch: int):
    
    model.train()
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)

    loss = F.nll_loss(output, target) # outputはone-hotの形式、targetは正解値の形式で比較。
    loss.backward()
    optimizer.step()

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=2, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    DataPath = "kihuData.txt"
    TargetPath = "kihuTargets.txt"

    Datas = []
    with open(DataPath) as f:
        
        for line in f:
            l = line.split(', ')
            data = []
            for i in l:
                data.append(float(i.rstrip(('\n'))))
            Datas.append(data)

    Targets = []
    with open(TargetPath) as f:
        for num in f:
            Targets.append(int(num.rstrip('\n')))

    datas = torch.tensor(Datas, dtype=torch.float32)
    targets = torch.tensor(Targets, dtype=torch.int64)
    datas = datas[: 10000]
    targets = targets[: 10000] # 10000件の学習。バッチ指定なしの状態では多すぎるとメモリ不足になる


    datas = datas.to(device)

    targets = targets.to(device)

    for epoch in range(1, args.epochs + 1):
        print(epoch)
        train(args, model, device, datas, targets, optimizer, epoch)
        scheduler.step()

    correct = 0
    trials = 140000
    for i in range(trials):
        data = torch.tensor([Datas[i]], dtype=torch.float32)
        data = data.to(device)

        ans = model.forward(data) # 現状のまま実用するなら、この値を上から順に合法手か探す。
        ansMax = ans.argmax(dim=1, keepdim=True)
        Tar = Targets[i]
        if(ansMax == Tar):
            correct += 1
    print('correct: ' + str(correct) + ' / ' + str(trials))
    print('Rating' + str(correct / trials))

    # x = index % 8
    # y = int(index / 8)
    #でインデックスとして取ってるデータを座標に変換可能。


if __name__ == '__main__':
    main()

