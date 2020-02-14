f = open('kihuFixedLite.txt')
lines = f.readlines()
f.close()
fixedLines = []
targets = []
for line in lines:
    l = line.split()
    fixed = [0] * 128
    target = (int(l[64]) - 1) * 8 + int(l[65]) - 1 
    if(l[66] == 'W'):
        for i in range(64):
            if(int(l[i]) == 1):
                fixed[i] = 1
            elif(int(l[i]) == 2):
                fixed[i + 64] = 1
    else:
        for i in range(64):
            if(int(l[i]) == 2):
                fixed[i] = 1
            elif(int(l[i]) == 1):
                fixed[i + 64] = 1
    '''チェック用
    
    print(fixed)

    board = [0] * 64

    for i in range(128):
        if(i < 64):
            if(fixed[i] == 1):
                board[i] = 1
        elif(i >= 64):
            if(fixed[i] == 1):
                board[i - 64] = 2
    for i in range(8):
        print(board[i * 8: i * 8 + 8])
    print(str(target % 8 + 1) + ", " + str(int(target / 8 + 1)))

    '''


    fixedLines.append(', '.join(map(str, fixed)))
    targets.append(str(target))
    
    


with open('kihuData.txt', mode= 'w') as f:
    f.write('\n'.join(fixedLines))
with open('kihuTargets.txt', mode= 'w') as f:
    f.write('\n'.join(targets))

