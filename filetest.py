import torch
DataPath = "kihuData.txt"
TargetPath = 'kihuTargets.txt'
'''
Datas = []
with open(DataPath) as f:
    
    for line in f:
        l = line.split(', ')
        data = []
        for i in l:
            data.append(float(i.rstrip(('\n'))))
        Datas.append(data)

'''
'''   
for t in Datas:
    print(t)
'''
Targets = []
with open(TargetPath) as f:
    for num in f:
        Targets.append(float(num.rstrip('\n')))   

#datas = torch.tensor(Datas, dtype=torch.float32)
targets = torch.tensor(Targets, dtype=torch.float32)
for t in targets:
    print(t)
