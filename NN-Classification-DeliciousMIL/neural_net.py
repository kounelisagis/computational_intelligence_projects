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
    x = F.tanh(self.fc1(x))
    return torch.sigmoid(self.fc2(x))


def train_nn(train_vectors, train_labels, test_vectors, test_labels, vocab_size):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Generate the model
    model = Net(input_layer_size=vocab_size, hidden_layer_size=int((vocab_size+20)/2), output_layer_size=20).to(device)
    model.train()

    # Generate the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)#, momentum=0.9)

    # criterion = nn.CrossEntropyLoss().to(device)
    criterion = nn.MSELoss().to(device)
    # criterion = Accuracy().to(device)

    results = []

    kf = KFold(n_splits=5)

    for fold, (train_index, validation_index) in enumerate(kf.split(train_vectors)):
        train_tensor_x = torch.Tensor(train_vectors[train_index]).to(device)
        train_tensor_y = torch.Tensor(train_labels[train_index]).to(device)
        # create the dataset
        trainset = TensorDataset(train_tensor_x, train_tensor_y)
        # create your dataloader
        trainloader = DataLoader(trainset, batch_size=100)

        validation_tensor_x = torch.Tensor(train_vectors[validation_index]).to(device)
        validation_tensor_y = torch.Tensor(train_labels[validation_index]).to(device)
        # create the dataset
        validationset = TensorDataset(validation_tensor_x, validation_tensor_y)

        objective(model, optimizer, criterion, vocab_size, trainloader, validationset, fold, results)

    # Test the model
    model.eval()

    # Turn off gradients computation
    with torch.no_grad():
        input_vectors, labels = torch.Tensor(test_vectors).to(device), torch.Tensor(test_labels).to(device)
        output = model(input_vectors)
        loss = criterion(output, labels)
        print('Loss:', loss.item())


    yaxis = np.array([result[-1] for result in results])
    plt.plot(yaxis)
    plt.show()


def objective(model, optimizer, criterion, vocab_size, trainloader, validationset, fold, results):

    EPOCHS = 10

    # Training of the model
    for epoch in range(EPOCHS):

        for batch_idx, (input_vectors, labels) in enumerate(trainloader):


            output = model(input_vectors)
            loss = criterion(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % int(len(trainloader)/10) == 0:
                # Validation of the model.
                model.eval()

                # Turn off gradients computation
                with torch.no_grad():
                    input_vectors, labels = validationset.tensors
                    output = model(input_vectors)
                    loss = criterion(output, labels)
                    
                    print('Fold:', fold, 'Epoch:', epoch, 'Batch:', batch_idx, 'Loss:', loss.item())
                    results.append( (fold, epoch, batch_idx, loss.item(), ) )

                model.train()
