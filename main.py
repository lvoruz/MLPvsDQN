import pNetwork
import deepQ
import gym
import numpy as np

'''
for testing purposes
mlpTest = pNetwork.mlp()
mlpTest.train(episodes = 100)
mlpTest.test()
mlpTest.exportData('mlpTest.txt')
'''
def writeData(data):
    with open('data.txt', 'w') as f:
        for i in range(len(data)):
            inputs = data[i]['input']
            outputs = data[i]['output']
            for j in range(len(inputs)):
                f.write(str(outputs[j]) + ' ')
                for k in range(len(inputs[j])):
                    if k < 3:
                        f.write(str(inputs[j][k]) + ' ')
                    else:
                        f.write(str(inputs[j][k]))
                f.write('\n')

def loadData():
    with open('data.txt') as f:
        lines = f.readlines()
    inputs = []
    outputs = []
    for line in lines:
        num = ''
        outputs.append(int(line[0]))
        states = []
        for i in range(2, len(line)):
            if line[i] == ' ':
                states.append(float(num))
                num = ''
            elif line[i] == '\n':
                states.append(float(num))
                num = ''
            else:
                num += line[i]
        inputs.append(states)
    return [{'input': inputs, 'output': outputs}]



def runFirstMlp():
    #train the first mlp with varying epoch limits
    mlp200 = pNetwork.mlp()
    mlp500 = pNetwork.mlp()
    mlp1000 = pNetwork.mlp()

    mlp200.train()#episodes = 200 by default
    mlp200.test()
    mlp200.exportData('mlp200.txt')

    mlp500.train(episodes = 500)
    mlp500.test()
    mlp500.exportData('mlp500.txt')

    mlp1000.train(episodes = 1000)
    mlp1000.test()
    mlp1000.exportData('mlp1000.txt')

def runDqn():
    avg, high, runs = deepQ.runDeepQ()
    with open('deepQ.txt', 'w') as f:
        f.write('Average Reward: ' + str(avg) + '\n')
        f.write('Highest Reward: ' + str(high))
    return runs


def runSecondMlp(runs):
    mlp = pNetwork.mlpVer2(runs)
    mlp.train()
    mlp.test()
    mlp.exportData('mlpVer2.txt')

dqRuns = None
while True:
    print('0. Quit\n1. First MLP\n2. DQN\n3. Second MLP\n4. All')
    i = input('Select: ')
    if i == '0':
        exit(0)
    elif i == '1':
        runFirstMlp()
    elif i == '2':
        dqRuns = runDqn()
        writeData(dqRuns)
    elif i == '3':
        if dqRuns:
            runSecondMlp(dqRuns)
        else:
            #load data from file
            dqRuns = loadData()
            runSecondMlp(dqRuns)
    elif i == '4':
        runFirstMlp()
        dqRuns = runDqn()
        runSecondMlp(dqRuns)
    else:
        print('invalid option try again.')
    
