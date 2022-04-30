import torch
from sklearn.model_selection import KFold
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np


class Net(nn.Module):

  def __init__(self, input_layer_size, hidden_layer_size, output_layer_size):
    super(Net, self).__init__()
    self.fc1 = nn.Linear(input_layer_size, hidden_layer_size)
    self.fc2 = nn.Linear(hidden_layer_size, output_layer_size)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    return torch.sigmoid(self.fc2(x))


def train_nn(train_vectors, train_labels, test_vectors, test_labels, vocab_size):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Generate the model
    model = Net(input_layer_size=vocab_size, hidden_layer_size=int((vocab_size+20)/2), output_layer_size=20).to(device)

    # Generate the optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.6, weight_decay=0.9)

    criterionBCE = nn.BCELoss().to(device)
    # criterion = nn.CrossEntropyLoss().to(device)
    criterionMSE = nn.MSELoss().to(device)

    results = []

    EPOCHS = 20

    # Training of the model
    for epoch in range(EPOCHS):

        kf = KFold(n_splits=5)

        for fold, (train_index, validation_index) in enumerate(kf.split(train_vectors)):
            train_tensor_x = torch.Tensor(train_vectors[train_index]).to(device)
            train_tensor_y = torch.Tensor(train_labels[train_index]).to(device)
            # create the dataset
            trainset = TensorDataset(train_tensor_x, train_tensor_y)
            # create your dataloader
            trainloader = DataLoader(trainset, batch_size=100, shuffle=True, drop_last=True)

            validation_tensor_x = torch.Tensor(train_vectors[validation_index]).to(device)
            validation_tensor_y = torch.Tensor(train_labels[validation_index]).to(device)
            # create the dataset
            validationset = TensorDataset(validation_tensor_x, validation_tensor_y)

            objective(model, optimizer, criterionBCE, trainloader, validationset, fold, epoch, results)


    # Final evaluation of the model
    model.eval()

    # Turn off gradients computation
    with torch.no_grad():
        input_vectors, labels = torch.Tensor(test_vectors).to(device), torch.Tensor(test_labels).to(device)
        output = model(input_vectors)
        lossBCE = criterionBCE(output, labels)
        lossMSE = criterionMSE(output, labels)
        acc = accuracy(output, labels)
        print('LossBCE:', lossBCE.item())
        print('LossMSE:', lossMSE.item())
        print('Accuracy:', acc.item())


    train_loss = np.array([result['train_loss'] for result in results])
    validation_loss = np.array([result['validation_loss'] for result in results])
    plt.title("Loss during training")
    plt.plot(train_loss, 'r', label="train loss")
    plt.plot(validation_loss, 'g', label="validation loss")
    plt.legend(loc="upper right")
    plt.show()

    train_acc = np.array([result['train_acc'] for result in results])
    validation_acc = np.array([result['validation_acc'] for result in results])
    plt.title("Accuracy during training")
    plt.plot(train_acc, 'r', label="train accuracy")
    plt.plot(validation_acc, 'g', label="validation accuracy")
    plt.legend(loc="lower right")
    plt.show()


def objective(model, optimizer, criterion, trainloader, validationset, fold, epoch, results):

    model.train()

    for batch_idx, (input_vectors, labels) in enumerate(trainloader):

        output = model(input_vectors)
        train_loss = criterion(output, labels)
        train_acc = accuracy(output, labels)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()


    model.eval()

    # Turn off gradients computation
    with torch.no_grad():
        input_vectors, labels = validationset.tensors
        output = model(input_vectors)
        validation_loss = criterion(output, labels)
        validation_acc = accuracy(output, labels)
        
        print('Epoch:', epoch, 'Fold:', fold, 'Train Loss:', train_loss.item(), 'Train Acc:', train_acc.item(), 'Validation Loss:', validation_loss.item(), 'Validation Acc:', validation_acc.item())
        results.append( {'epoch': epoch, 'fold': fold, 'train_loss': train_loss.item(), 'train_acc': train_acc.item(), 'validation_loss': validation_loss.item(), 'validation_acc': validation_acc.item()} )


def accuracy(predictions, target):
    predictions = predictions.round().int()
    target = target.int()

    acc = (torch.bitwise_and(predictions, target).sum() / torch.bitwise_or(predictions, target).sum())# / len(target)
    return acc
