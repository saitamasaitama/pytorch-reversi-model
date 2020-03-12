# データは
#http://meipuru-344.hatenablog.com/entry/2017/11/27/205448
# ここから取得。
# 以下の処理でさらに使いやすいように変換する。

f = open('kihuFixed.txt')
lines = f.readlines()
f.close()
fixedLines = []
targets = []
for line in lines:
    l = line.split()
    boardData = [[0 for i in range(8)] for j in range (8)]
    target = (int(l[64]) - 1) * 8 + int(l[65]) - 1 
    if(l[66] == 'W'):
        me = 1
        you = 2
    else:
        me = 2
        you = 1
    for i in range(64):
        if(int(l[i]) == 1):
            boardData[int(i / 8)][int(i % 8)] = 1
        elif(int(l[i]) == 2):
            boardData[int(i / 8)][int(i % 8)] = -1
    print(boardData)
    for b in boardData:
        fixedLines.append(', '.join(map(str, b)))
        #print(b)
    targets.append(str(target))
    
    


with open('kihuData2.txt', mode= 'w') as f:
    f.write('\n'.join(fixedLines))
with open('kihuTargets2.txt', mode= 'w') as f:
    f.write('\n'.join(targets))

