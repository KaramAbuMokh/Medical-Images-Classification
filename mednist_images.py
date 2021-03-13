import numpy as np
import os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as om
import torchvision as tv

# confusion matrix plotting
from sklearn.metrics import confusion_matrix, accuracy_score
from mlxtend.plotting import plot_confusion_matrix

if torch.cuda.is_available():  # Make sure GPU is available
    dev = torch.device("cuda:0")
    kwar = {'num_workers': 8, 'pin_memory': True}
    cpu = torch.device("cpu")
else:
    print("Warning: CUDA not found, CPU only.")
    dev = torch.device("cpu")
    kwar = {}
    cpu = torch.device("cpu")

np.random.seed(551)

# ----------    get the names of the files and array of the classes   ----------------

dataDir = 'data'  # The main data directory
classNames = os.listdir(dataDir)  # Each type of image can be found in its own subdirectory
numClass = len(classNames)  # Number of types = number of subdirectories
imageFiles = [[os.path.join(dataDir, classNames[i], x) for x in os.listdir(os.path.join(dataDir, classNames[i]))]
              for i in range(numClass)]  # A nested list of filenames
numEach = [len(imageFiles[i]) for i in range(numClass)]  # A count of each type of image
imageFilesList = []  # Created an un-nested list of filenames
imageClass = []  # The labels -- the type of each individual image in the list
for i in range(numClass):
    imageFilesList.extend(imageFiles[i])
    imageClass.extend([i] * numEach[i])
numTotal = len(imageClass)  # Total number of images
imageWidth, imageHeight = Image.open(imageFilesList[0]).size  # The dimensions of each image

print("There are", numTotal, "images in", numClass, "distinct categories")
print("Label names:", classNames)
print("Label counts:", numEach)
print("Image dimensions:", imageWidth, "x", imageHeight)

# -------------------   scale the images   -----------------------------

toTensor = tv.transforms.ToTensor()


def scaleImage(x):  # Pass a PIL image, return a tensor
    y = toTensor(x)
    if (y.min() < y.max()):  # Assuming the image isn't empty, rescale so its values run from 0 to 1
        y = (y - y.min()) / (y.max() - y.min())
    z = y - y.mean()  # Subtract the mean value of the image
    return z


# -----------------------   read the files of the images   ---------------------------


imageTensor = torch.stack(
    [scaleImage(Image.open(x)) for x in imageFilesList])  # Load, scale, and stack image (X) tensor
classTensor = torch.tensor(imageClass)  # Create label (Y) tensor
print("Rescaled min pixel value = {:1.3}; Max = {:1.3}; Mean = {:1.3}"
      .format(imageTensor.min().item(), imageTensor.max().item(), imageTensor.mean().item()))

# -----------------------   Partitioning into Training, Validation, and Testing Sets.  ---------------------------

validFrac = 0.2 # Define the fraction of images to move to validation dataset
testFrac = 0.2 # Define the fraction of images to move to test dataset

trainList = []
validList = []
testList = []

for i in range(numTotal):
    rann = np.random.random()  # Randomly reassign images
    if rann < validFrac:
        validList.append(i)
    elif rann < testFrac + validFrac:
        testList.append(i)
    else:
        trainList.append(i)

nTrain = len(trainList)  # Count the number in each set
nValid = len(validList)
nTest = len(testList)
print("Training images =", nTrain, "Validation =", nValid, "Testing =", nTest)

trainIds = torch.tensor(trainList)  # Slice the big image and label tensors up into
validIds = torch.tensor(validList)  # training, validation, and testing tensors
testIds = torch.tensor(testList)
trainX = imageTensor[trainIds, :, :, :]
trainY = classTensor[trainIds]
validX = imageTensor[validIds, :, :, :]
validY = classTensor[validIds]
testX = imageTensor[testIds, :, :, :]
testY = classTensor[testIds]


# -------------------------- model class -----------------------------------

class MedNet(nn.Module):
    def __init__(self, xDim, yDim, numC):  # Pass image dimensions and number of labels when initializing a model
        super(MedNet, self).__init__()  # Extends the basic nn.Module to the MedNet class
        # The parameters here define the architecture of the convolutional portion of the CNN. Each image pixel
        # has numConvs convolutions applied to it, and convSize is the number of surrounding pixels included
        # in each convolution. Lastly, the numNodesToFC formula calculates the final, remaining nodes at the last
        # level of convolutions so that this can be "flattened" and fed into the fully connected layers subsequently.
        # Each convolution makes the image a little smaller (convolutions do not, by default, "hang over" the edges
        # of the image), and this makes the effective image dimension decreases.

        numConvs1 = 5
        convSize1 = 7
        numConvs2 = 10
        convSize2 = 7
        numNodesToFC = numConvs2 * (xDim - (convSize1 - 1) - (convSize2 - 1)) * (
                yDim - (convSize1 - 1) - (convSize2 - 1))

        # nn.Conv2d(channels in, channels out, convolution height/width)
        # 1 channel -- grayscale -- feeds into the first convolution. The same number output from one layer must be
        # fed into the next. These variables actually store the weights between layers for the model.

        self.cnv1 = nn.Conv2d(1, numConvs1, convSize1)
        self.cnv2 = nn.Conv2d(numConvs1, numConvs2, convSize2)

        # These parameters define the number of output nodes of each fully connected layer.
        # Each layer must output the same number of nodes as the next layer begins with.
        # The final layer must have output nodes equal to the number of labels used.

        fcSize1 = 400
        fcSize2 = 80

        # nn.Linear(nodes in, nodes out)
        # Stores the weights between the fully connected layers

        self.ful1 = nn.Linear(numNodesToFC, fcSize1)
        self.ful2 = nn.Linear(fcSize1, fcSize2)
        self.ful3 = nn.Linear(fcSize2, numC)

    def forward(self, x):
        # This defines the steps used in the computation of output from input.
        # It makes uses of the weights defined in the __init__ method.
        # Each assignment of x here is the result of feeding the input up through one layer.
        # Here we use the activation function elu, which is a smoother version of the popular relu function.

        x = F.elu(self.cnv1(x))  # Feed through first convolutional layer, then apply activation
        x = F.elu(self.cnv2(x))  # Feed through second convolutional layer, apply activation
        x = x.view(-1, self.num_flat_features(x))  # Flatten convolutional layer into fully connected layer
        x = F.elu(self.ful1(x))  # Feed through first fully connected layer, apply activation
        x = F.elu(self.ful2(x))  # Feed through second FC layer, apply output
        x = self.ful3(x)  # Final FC layer to output. No activation, because it's used to calculate loss
        return x

    def num_flat_features(self, x):  # Count the individual nodes in a layer
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


model = MedNet(imageWidth, imageHeight, numClass).to(dev)

# -------------------------    training the model    -------------------------------------------

learnRate = 0.01  # Define a learning rate.
maxEpochs = 30  # Maximum training epochs
t2vRatio = 2.5  # Maximum allowed ratio of validation to training loss
t2vEpochs = 3  # Number of consecutive epochs before halting if validation loss exceeds above limit
batchSize = 4  # Batch size. Going too large will cause an out-of-memory error.
trainBats = nTrain // batchSize  # Number of training batches per epoch. Round down to simplify last batch
validBats = nValid // batchSize  # Validation batches. Round down
testBats = -(-nTest // batchSize)  # Testing batches. Round up to include all
CEweights = torch.zeros(numClass)  # This takes into account the imbalanced dataset.
for i in trainY.tolist():  # By making rarer images count more to the loss,
    CEweights[i].add_(1)  # we prevent the model from ignoring them.
CEweights = 1. / CEweights.clamp_(min=1.)  # Weights should be inversely related to count
CEweights = (CEweights * numClass / CEweights.sum()).to(dev)  # The weights average to 1
opti = om.SGD(model.parameters(), lr=learnRate)  # Initialize an optimizer

for i in range(maxEpochs):
    model.train()  # Set model to training mode
    epochLoss = 0.
    permute = torch.randperm(nTrain)  # Shuffle data to randomize batches
    trainX = trainX[permute, :, :, :]
    trainY = trainY[permute]
    for j in range(trainBats):  # Iterate over batches
        opti.zero_grad()  # Zero out gradient accumulated in optimizer
        batX = trainX[j * batchSize:(j + 1) * batchSize, :, :, :].to(dev)  # Slice shuffled data into batches
        batY = trainY[j * batchSize:(j + 1) * batchSize].to(dev)  # .to(dev) moves these batches to the GPU
        yOut = model(batX)  # Evalute predictions
        loss = F.cross_entropy(yOut, batY, weight=CEweights)  # Compute loss
        epochLoss += loss.item()  # Add loss
        loss.backward()  # Backpropagate loss
        opti.step()  # Update model weights using optimizer
    validLoss = 0.
    permute = torch.randperm(nValid)  # We go through the exact same steps, without backprop / optimization
    validX = validX[permute, :, :, :]  # in order to evaluate the validation loss
    validY = validY[permute]
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Temporarily turn off gradient descent
        for j in range(validBats):
            opti.zero_grad()
            batX = validX[j * batchSize:(j + 1) * batchSize, :, :, :].to(dev)
            batY = validY[j * batchSize:(j + 1) * batchSize].to(dev)
            yOut = model(batX)
            validLoss += F.cross_entropy(yOut, batY, weight=CEweights).item()
    epochLoss /= trainBats  # Average loss over batches and print
    validLoss /= validBats
    print("Epoch = {:-3}; Training loss = {:.4f}; Validation loss = {:.4f}".format(i, epochLoss, validLoss))
    if validLoss > t2vRatio * epochLoss:
        t2vEpochs -= 1  # Test if validation loss exceeds halting threshold
        if t2vEpochs < 1:
            print("Validation loss too high; halting to prevent overfitting")
            break


torch.save(model.state_dict(), 'model')
# ------------  evaluate the model on the test set -----------------------------
model = MedNet(imageWidth, imageHeight, numClass).to(dev)
model.load_state_dict(torch.load('model'))
model.eval()
with torch.no_grad():
    permute = torch.randperm(nTest)  # Shuffle test data
    testX = testX[permute, :, :, :]
    testY = testY[permute]
    pred = []

    for j in range(testBats):  # Iterate over test batches
        batX = testX[j * batchSize:(j + 1) * batchSize, :, :, :].to(dev)
        batY = testY[j * batchSize:(j + 1) * batchSize].to(dev)
        yOut = model(batX)  # Pass test batch through model
        predict = yOut.max(1)[1].cpu().numpy()
        pred = np.concatenate((pred, predict))

class_names = ['BreastMRI', 'Hand', 'HeadCT', 'CXR', 'ChestCT', 'AbdomenCT']
accuracy = accuracy_score(pred, testY)
print("Accuracy = ", accuracy)
cm = confusion_matrix(pred, testY)
_ = plot_confusion_matrix(cm, colorbar=True, class_names=class_names)
