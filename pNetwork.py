import numpy as np
import gym

def sigmoid(sums):
    return 1/(1 + np.exp(-sums))

def sigprime(sig):
    return sig * (1 - sig)

def computeAccuracy(conMatrix):
    right = 0
    wrong = 0
    for i in range(len(conMatrix)):
        r = 0
        w = 0
        for j in range(len(conMatrix[i])):
            if i == j:
                r = conMatrix[i][j]
            else:
                w += conMatrix[i][j]
        right += r
        wrong += w
    return (right/(right + wrong)) * 100

class mlp():
    def __init__(self):
        #make environment
        self.env = gym.make('CartPole-v0')
        #set num inputs, hidden layers and outputs
        self.inputs = 4
        self.hunits = 3
        self.outputs = 2
        #set up weights
        self.weights = []
        #weights from input to hidden layer
        self.weights.append(np.random.uniform(low = -.05, high = .05, size =(self.inputs+1, self.hunits)))
        #weights from hidden layer to output layer
        self.weights.append(np.random.uniform(low = -.05, high = .05, size =(self.hunits+1, self.outputs)))
        #bias values
        self.bias = [1., 1.]
        #learning rate
        self.eta = .1
        #momentum
        self.alpha = .9
        #network average reward
        self.avgRwd = 0.0
        #network highest reward
        self.highRwd = 0.0
        self.trainingEpochs = 0
        #confusion matrix
        #self.conMatrix = np.zeros((2,2), dtype=int)

    '''
    training function for mlp
    '''
    def train(self, tExamples = 200, episodes = 200):
        #init epoch
        epoch = 0
        #init weights
        updateWeights = []
        for i in range(len(self.weights)):
            updateWeights.append(np.zeros(np.shape(self.weights[i])))
        print('Beginning Training...')

        while epoch < episodes:
            print('Epoch: ' + str(epoch + 1))
            #get initial state
            state = np.array(self.env.reset())
            totalReward = 0
            for i in range(tExamples):
                #insert bias node
                state = np.insert(state, 0, self.bias[0], axis = 0)
                #print(np.shape(state))
                #pass state through layers to get action
                ihsums = np.dot(state, self.weights[0])
                hActivations = sigmoid(ihsums)
                hActivations = np.insert(hActivations, 0, self.bias[1], axis = 0)
                hosums = np.dot(hActivations, self.weights[1])
                oActivations = sigmoid(hosums)
                #calculate action based off activation function
                action = np.argmax(oActivations)#1 pushes to right, 0 to left
                #take step
                nextState, reward, done, info = self.env.step(action)
                totalReward += reward
                #render
                self.env.render()
                #determine target by checking pole orientation of previous state, if + then go right, - is left
                pole_angle = state[2]
                targets = np.full(self.outputs, .1)
                if pole_angle < 0:
                    targets[0] = .9
                else:
                    targets[1] = .9
                #update weights
                #delta calc
                deltak = sigprime(oActivations) * (targets - oActivations)
                deltaj = sigprime(hActivations) * (np.dot(deltak, np.transpose(self.weights[1])))
                #h->output weights
                updateWeights[1] = self.eta*(np.dot(np.transpose(np.reshape(hActivations,(1,-1))),
                np.reshape(deltak,(1,-1)))) + self.alpha * updateWeights[1]
                #input=>h weights
                updateWeights[0] = self.eta*(np.dot(np.transpose(np.reshape(state, (1,-1))),
                np.reshape(np.delete(deltaj,0),(1,-1)))) + self.alpha*updateWeights[0]
                for i in range(len(self.weights)):
                    self.weights[i] += updateWeights[i]
                state = nextState
                if(done):#early termination
                    break#go to next epoch
                
            #update epoch
            epoch += 1
            self.trainingEpochs += 1
            if totalReward == 200:#perfect run
                break#earlytermination
        #close env
        self.env.close()
    
    '''
    Testing function for mlp
    '''
    def test(self, tExamples = 200, episodes = 100):
        epoch = 0
        print('Beginning Testing')
        while epoch < episodes:
            print('Epoch: ' + str(epoch + 1))
            totalReward = 0.0
            state = np.array(self.env.reset())
            for i in range(tExamples):
                state = np.insert(state, 0, self.bias[0], axis=0)
                ihsums = np.dot(state, self.weights[0])
                hActivations = sigmoid(ihsums)
                hActivations = np.insert(hActivations, 0, self.bias[1], axis = 0)
                hosums = np.dot(hActivations, self.weights[1])
                oActivations = sigmoid(hosums)
                action = np.argmax(oActivations)
                state, reward, done, info = self.env.step(action)
                totalReward += reward
                self.env.render()
                if(done):#early termination
                    break
            if totalReward > self.highRwd:
                self.highRwd = totalReward
            self.avgRwd += totalReward
            epoch += 1
        self.avgRwd /= episodes
        self.env.close()
        return

    def exportData(self, fname):
        with open(fname, 'w') as f:
            f.write('Epochs trained: ' + str(self.trainingEpochs) + '\n')
            f.write('Average Reward: ' + str(self.avgRwd) + '\n')
            f.write('Highest Reward: ' + str(self.highRwd))
                
class mlpVer2():
    def __init__(self, data):
        #make environment
        self.env = gym.make('CartPole-v0')
        #copy data
        self.data = data
        #set num inputs, hidden layers and outputs
        self.inputs = 4
        self.hunits = 3
        self.outputs = 2
        #set up weights
        self.weights = []
        #weights from input to hidden layer
        self.weights.append(np.random.uniform(low = -.05, high = .05, size =(self.inputs+1, self.hunits)))
        #weights from hidden layer to output layer
        self.weights.append(np.random.uniform(low = -.05, high = .05, size =(self.hunits+1, self.outputs)))
        #bias values
        self.bias = [1., 1.]
        #learning rate
        self.eta = .1
        #momentum
        self.alpha = .9
        #network average reward
        self.avgRwd = 0.0
        #network highest reward
        self.highRwd = 0.0
        self.trainingEpochs = 0
        self.conMatrix = np.zeros((self.outputs, self.outputs), dtype=int)
    
    def splitData(self):
        #data is in a list of 100 dictionaries where data[i]['input'] is a list of states and
        #data[i]['output'] is a list of actions. The state data[i]['input'][j] corresponds
        #to data[i]['output'][j]
        trainValues = []
        targets = []
        for i in range(len(self.data)):#should be 100
            inputs = self.data[i]['input']
            outputs = self.data[i]['output']
            for j in range(len(inputs)):
                trainValues.append(inputs[j])#1x4 array

                targets.append(outputs[j])#singular value
        return np.array(trainValues), np.array(targets)

    def train(self, episodes = 500):
        trainValues, targets = self.splitData()
        epoch = 0
        updateWeights = []
        accHist = [0.0]
        for i in range(len(self.weights)):
            updateWeights.append(np.zeros(np.shape(self.weights[i])))
        print('Beginning Training')
        while epoch < episodes:
            print('Epoch: ' + str(epoch + 1))
            self.conMatrix = np.zeros((self.outputs, self.outputs), dtype=int)
            for i in range(len(trainValues)):
                state = np.insert(trainValues[i], 0, self.bias[0], axis = 0)
                ihsums = np.dot(state, self.weights[0])
                hActivations = sigmoid(ihsums)
                hActivations = np.insert(hActivations, 0, self.bias[1], axis = 0)
                hosums = np.dot(hActivations, self.weights[1])
                oActivations = sigmoid(hosums)
                action = np.argmax(oActivations)
                target = targets[i]
                self.conMatrix[target][action] += 1
                t = np.full(self.outputs, .1)
                t[target] = .9
                deltak = sigprime(oActivations) * (t - oActivations)
                deltaj = sigprime(hActivations) * (np.dot(deltak, np.transpose(self.weights[1])))
                updateWeights[1] = self.eta*(np.dot(np.transpose(np.reshape(hActivations, (1,-1))),
                np.reshape(deltak, (1,-1)))) + self.alpha * updateWeights[1]
                updateWeights[0] = self.eta*(np.dot(np.transpose(np.reshape(state, (1,-1))),
                np.reshape(np.delete(deltaj,0),(1,-1)))) + self.alpha*updateWeights[0]
                for i in range(len(self.weights)):
                    self.weights[i] += updateWeights[i]
            epoch += 1
            accHist.append(computeAccuracy(self.conMatrix))
            if len(accHist) > 2:
                diff = accHist[-1] - accHist[-2]
                if diff <.01 and diff >-.01:
                    print('training done!')
                    return

    def test(self, tExamples = 200, episodes = 100):
        epoch = 0
        print('Beginning Testing')
        while epoch < episodes:
            print('Epoch ' + str(epoch + 1))
            totalReward = 0.0
            state = np.array(self.env.reset())
            for i in range(tExamples):
                state = np.insert(state, 0, self.bias[0], axis = 0)
                ihsums = np.dot(state, self.weights[0])
                hActivations = sigmoid(ihsums)
                hActivations = np.insert(hActivations, 0, self.bias[1], axis = 0)
                hosums = np.dot(hActivations, self.weights[1])
                oActivations = sigmoid(hosums)
                action = np.argmax(oActivations)
                state, reward, done, info = self.env.step(action)
                totalReward += reward
                self.env.render()
                if(done):
                    break
            if totalReward > self.highRwd:
                self.highRwd = totalReward
            self.avgRwd += totalReward
            epoch += 1
        self.avgRwd /= episodes
        self.env.close()

    def exportData(self, fname):
        with open(fname, 'w') as f:
            f.write('Average Reward: ' + str(self.avgRwd) + '\n')
            f.write('Highest Reward: ' + str(self.highRwd))