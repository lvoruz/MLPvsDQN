from os.path import dirname, join
import csv
import numpy as np
#import random
#from datetime import datetime

'''
mlp training
1. present input to input layer
    n + 1 inputs for bias (785)

2. forward to hidden layer
    n + 1 nodes in hidden layer (785)
    hj = sigmoid(sum(wji*xi+wj0))
    
3. forward to output layer
    m nodes in output layer (10)
    ok = sigmoid(sum(wkj*hj + wk0))

4. determine error

5. back-propagate to update weights
'''
def createCSV(fileName, data):
    currentDir = dirname(__file__)
    path = join(currentDir, fileName)
    with open(path, 'w', newline='') as new_file:
        csv_writer = csv.writer(new_file, delimiter=',')
        for i in range(0, len(data)):
            csv_writer.writerow((i, data[i][0]))
        for i in range(0,len(data[-1][1])): 
            csv_writer.writerow(data[-1][1][i])

def loadData(fileName, bias):
    print('Loading ' + fileName + '...')
    #get file path
    currentDir = dirname(__file__)
    path = join(currentDir, fileName)
    #initialize set
    targets = []
    inputs = []
    #open file
    with open(path) as csvfile:
        #read the file
        readCSV = csv.reader(csvfile, delimiter=',')
        #split data, first column is label, rest are 0-255 grayscale values
        #skip first row, label names
        next(readCSV)
        for row in readCSV:
            targets.append(int(row[0]))
            values = [bias]
            for i in range(1, len(row)):
                values.append(float(row[i])/255)
            inputs.append(values)
    print('Loading complete!')
    return np.array(inputs), targets

'''
Takes the sums array (which is an array of values that as xw for all inputs/weights)
and then performs the operation 1/(1+e^-x) on all elements using the np.exp function
'''
def sigmoid(sums):
    return 1/(1 + np.exp(-sums))

#derivative of sigmoid given that sigmoid has already been applied to the array
def sigPrime(sig):
    return sig * (1 - sig)

#computes accuracy of mlp from its confusion matrix
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

'''
initialize for prog 1: p = pNetwork.mlp(784, n, 10, eta)
where n is the number of units in the hidden layer and eta is the learning rate
other values (individual bias values, threshold) are editable, but default to
1 for each bias and 0 for threshold.
'''

class mlp():
    def __init__(self, inputs, hunits, outputs, eta, alpha, ibias = 1., hbias = 1.):
        #random.seed(datetime.now())
        self.inputs = inputs
        self.hunits = hunits
        self.outputs = outputs
        '''
        weights between inputs and hidden layer
        numbers are generated within np.random
        '''
        self.hweights = np.random.uniform(low=-.05, high=.05, size=(inputs+1, hunits))
        #weights between hidden layer and output layer
        self.oweights = np.random.uniform(low=-.05, high=.05, size=(hunits+1, outputs))
        #print(self.oweights)
        #bias node values
        self.ibias = ibias
        self.hbias = hbias
        #learning rate
        self.eta = eta
        #momentum value
        self.alpha = alpha
        #accuracy of network (starts at 0)
        self.accuracy = 0.0
        #confusion matrix for data in report
        self.conMatrix = np.zeros((outputs, outputs), dtype=int)
        self.testConMatrix = np.zeros((outputs, outputs), dtype=int)

    def train(self, tExamples = None, episodes = 50):
        #load the training set and target values
        trainingValues, targetValues = loadData('mnist_train.csv', self.ibias)
        testingValues, testTargetValues = loadData('mnist_test.csv', self.ibias)
        if tExamples == None:
            tExamples = len(trainingValues)
        else:
            if tExamples > len(trainingValues):
                print('input example value exceeds capacity')
                exit(1)
        #initialize epoch and accuracy history for data collection
        epoch = 0
        accHist = [(0.0, self.conMatrix)]
        testHist = [(0.0, self.conMatrix)]
        #initialize initial weight change for update function
        updateOweights = np.zeros(np.shape(self.oweights), dtype=float)
        updateHweights = np.zeros(np.shape(self.hweights), dtype=float)
        #start loop
        print('Begining training...')
        #percent = 0
        #print('percentage: 0%, ', end='')
        while epoch < episodes:
            #print(str(epoch + 1), end = ', ')
            '''
            if epoch%5 == 0:
                print(percent, end='%, ')
                percent += 10
            '''
            for i in range(tExamples):
                '''
                get value in correct shape
                sum up the input weights by dotting trainingValues[i] (a 1x785 matrix) with hweights
                (a 785xN, N = number of hidden units) to get a 1xN matrix
                '''
                #np.reshape(trainingValues[i], (1,-1))
                ihsums = np.dot(trainingValues[i], self.hweights)
                '''
                now we have a 1xN matrix, now we need to compute activations with the sigmoid function
                and add a 1 to to the front end of the matrix to account for the bias node
                '''
                hActivations = sigmoid(ihsums)
                #print(hActivations)
                hActivations = np.insert(hActivations, 0, self.hbias, axis=0)
                #print(hActivations)
                '''
                matrix is now 1xN+1 to account for bias node in hidden layer, now we repeat the process
                for the output layer, except we don't need to append a bias node
                '''
                hosums = np.dot(hActivations, self.oweights)
                oActivations = sigmoid(hosums)
                #make predictionn and update confusion matrix
                prediction = np.argmax(oActivations)
                self.conMatrix[targetValues[i]][prediction] += 1
                '''
                matrix is now 1xM, where M is the number of outputs (in this case 10). Now we need to make
                a 0 vector of size (self.outputs) and change the targetValues[i] index to 1 to represent
                our target values
                '''
                targets = np.full(self.outputs, .1)
                targets[targetValues[i]] = .9
                '''
                Now we calculate error terms for output
                deltak <- ok(1-ok)(tk-ok)
                '''
                #deltak = (targets - oActivations) * oActivations * (1. - oActivations)
                deltak = sigPrime(oActivations) * (targets - oActivations)
                '''
                now we do it for the hidden units
                deltaj <- hj(1-hj)(sum(wkj * deltak)) kEoutput units
                '''
                deltaj = sigPrime(hActivations) * (np.dot(deltak,np.transpose(self.oweights)))
                '''
                print(deltak)
                print(deltaj)
                return
                '''
                #now we compute weight updates
                #print(np.shape(np.transpose(oActivations)))
                ##print(np.shape(deltak))
                updateOweights = self.eta*(np.dot(np.transpose(np.reshape(hActivations,(1,-1))), 
                np.reshape(deltak,(1,-1)))) + self.alpha*updateOweights
                #print(np.shape(updateOweights))
                #print(updateOweights)
                updateHweights = self.eta*(np.dot(np.transpose(np.reshape(trainingValues[i],(1,-1))),
                np.reshape(np.delete(deltaj, 0),(1,-1)))) + self.alpha*updateHweights
                
                '''
                print(updateHweights)
                print(updateOweights)
                return
                '''
                '''
                print(self.oweights)
                print(updateOweights)
                print(updateHweights)
                '''
                self.oweights += updateOweights
                self.hweights += updateHweights
            #test the network now
            for i in range(len(testingValues)):
                ihsums = np.dot(testingValues[i], self.hweights)
                hActivations = sigmoid(ihsums)
                hActivations = np.insert(hActivations, 0, self.hbias, axis=0)
                hosums = np.dot(hActivations, self.oweights)
                oActivations = sigmoid(hosums)
                #make predictionn and update confusion matrix
                prediction = np.argmax(oActivations)
                self.testConMatrix[testTargetValues[i]][prediction] += 1

            #update epoch
            epoch += 1
            #update accuracy and accHist
            self.accuracy = computeAccuracy(self.testConMatrix)
            accHist.append((computeAccuracy(self.conMatrix), np.array(self.conMatrix, copy=True)))
            testHist.append((self.accuracy, np.array(self.testConMatrix, copy=True)))
            print('epoch: ' + str(epoch) + '\ntraining accuracy: ' + str(accHist[-1][0]) + '\ntraining confusion matrix:')
            print(self.conMatrix)
            print('testing accuracy: ' + str(self.accuracy) + '\ntesting confusion matrix:')
            print(self.testConMatrix)
            #check stop condition
            if len(accHist) > 2:
                diff = accHist[-1][0] - accHist[-2][0]
                if diff < 0.01 and diff > -0.01:#difference between most recent accuracies is less than .01%
                    print('Training done!')
                    print('Epochs: ' + str(epoch))
                    print('Final Accuracy: ' + str(self.accuracy) + '%')
                    return accHist, testHist
                
            #clear conMatrix for next epoch
            self.conMatrix = np.zeros((self.outputs, self.outputs), dtype=int)
            self.testConMatrix = np.zeros((self.outputs, self.outputs), dtype=int)
        #training done, return the accHist, output the data and begin testing
        print('Training Done!')
        print('Epochs: ' + str(epoch))
        print('Final Accuracy: ' + str(self.accuracy) + '%')
        print(accHist[-1][1])
        return accHist, testHist