import torch
from torch import nn
import numpy as np
import sklearn as sk
from sklearn.model_selection import train_test_split


def final_func(x):
    one = max(x)
    arr = []
    for i in range(0, len(x)):
        if x[i] == one:
           arr.append(1.0)
        else:
            arr.append(0.0)
    tensor = torch.from_numpy(np.array(arr))
    return tensor

def ff_func(X):
    lst = []
    for i in X:
        tensor = final_func(i)
        lst.append(tensor)

    res = torch.tensor([])
    for i in lst:
        i = torch.tensor(i)
        res = torch.cat((res,i), dim=0)
    res = res.reshape((len(lst), 3))
    return res



"""
part1 = 3333
part2 = 3334


#X = (torch.rand(size=(1000,42)) < 0.25).int()
#This is creating an X which consists of 333 arrays of 42 1's, 333 arrays of 42 2's, 334 arrays of 42 3's
in1 = [1] * 42
in2 = [2] * 42
in3 = [3] * 42
X = [in1] * part1
X.extend([in2] * part1)
X.extend([in3] * part2)
X = torch.from_numpy(np.array(X)).float()
arr1 = [[0,0,1]] * part1
arr2 = [[0,1,0]] * part1
arr3 = [[1,0,0]] * part2
arr1.extend(arr2)
arr1.extend(arr3)
y = np.array(arr1)

X = X.float()
y = torch.from_numpy(y).float()
"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=42)


#print(X)
training_data = list(zip(X_train, y_train))
test_data = list(zip(X_test, y_test))


from torch.utils.data import DataLoader
BATCHSIZE = 100
# We want a nn with 42 input neurons, 22 neurons for one hidden layer, 3 nuerons per output layer
model = torch.nn.Sequential(
    torch.nn.Linear(42, 22),
    torch.nn.ReLU(),
    torch.nn.Linear(22, 3),
    torch.nn.Softmax(),
)

optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=0.93)
criterion = nn.CrossEntropyLoss()
model.train()
train_dataloader = DataLoader(training_data, batch_size=BATCHSIZE)
for epoch in range(5):
    for batch_num, (X, y) in enumerate(train_dataloader):
        # Compute prediction and loss
        pred = model(X)
        print(pred)
        #pred = [final_func(x) for x in pred]
        from torch.autograd import Variable
        loss = criterion(pred, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_num % 100 == 0:
            acc = (torch.sum(torch.argmax(pred, dim=0) == y).item()) / len(y)
            print("Epoch: {}.\tBatch: {}.\tTrain accuracy: {:.4f}.\tLoss: {:.4f}".format(epoch, batch_num, acc, loss.item()))




test_dataloader = DataLoader(test_data, batch_size=BATCHSIZE)

size = len(test_dataloader.dataset)
num_batches = len(test_dataloader)
test_loss, acc = 0, 0

model.eval()

with torch.no_grad():
    acc = 0
    for X, y in test_dataloader:
        pred = model(X)
        pred = ff_func(pred)
        test_loss += criterion(pred, y).item()
        #I have to convert the softmax probabilities into an input that is of the form xxy, xyx, or yxx where y =1 and x =0.
        #acc += (torch.sum(torch.argmax(pred, dim=0) == y).item())
        #acc += (pred == y).sum()
        for i in range(0, len(pred)):
            if torch.equal(pred[i], y[i].double()):
                acc += 1

test_loss /= num_batches
acc /= size
print("Test accuracy: {:.4f}. Test loss: {:.4f}".format(acc, test_loss))