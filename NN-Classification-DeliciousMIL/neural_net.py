import torch
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


class Net(nn.Module):

  def __init__(self, n_features, hidden_layer_size):
    super(Net, self).__init__()
    self.fc1 = nn.Linear(n_features, hidden_layer_size)
    self.fc2 = nn.Linear(hidden_layer_size, 20)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    return torch.sigmoid(self.fc2(x))


def create_nn(train_vectors, train_labels, test_vectors, test_labels, vocab_size):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    kf = KFold(n_splits=5)


    for fold, (train_index, validation_index) in enumerate(kf.split(train_vectors)):
        train_tensor_x = torch.Tensor(train_vectors[train_index]).to(device)
        train_tensor_y = torch.Tensor(train_labels[train_index]).to(device)
        # create the dataset
        trainset = TensorDataset(train_tensor_x,train_tensor_y)
        # create your dataloader
        trainloader = DataLoader(trainset, batch_size=10)

        validation_tensor_x = torch.Tensor(train_vectors[validation_index]).to(device)
        validation_tensor_y = torch.Tensor(train_labels[validation_index]).to(device)
        # create the dataset
        validationset = TensorDataset(validation_tensor_x,validation_tensor_y)
        # create your dataloader
        validationloader = DataLoader(validationset, batch_size=100)

        objective(vocab_size, trainloader, validationloader)



def objective(vocab_size, trainloader, validationloader):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Generate the model
    model = Net( vocab_size, int((8520+20)/2) ).to(device)

    # Generate the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)#, momentum=0.9)

    criterion = nn.CrossEntropyLoss().to(device)

    EPOCHS = 20
    losses = []
    checkpoint_losses = []
    n_total_steps = len(trainloader)

    # Training of the model
    for epoch in range(EPOCHS):
        model.train()

        for batch_idx, (input_vectors, labels) in enumerate(trainloader):

            output = model(input_vectors)
            loss = criterion(output, labels)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (batch_idx+1) % (int(n_total_steps/1)) == 0:
                checkpoint_loss = torch.tensor(losses).mean().item()
                checkpoint_losses.append(checkpoint_loss)
                print (f'Epoch [{epoch+1}/{EPOCHS}], Step [{batch_idx+1}/{n_total_steps}], Loss: {checkpoint_loss:.4f}')


        # Validation of the model.
        model.eval()

        # turn off gradients computation
        with torch.no_grad():
            model_result = []
            targets = []
            for batch_idx, (images, labels) in enumerate(validationloader):
                output = model(images)
                model_result.extend(output.cpu().numpy())
                targets.extend(labels.cpu().numpy())

        report = calculate_metrics(np.array(model_result), np.array(targets))
        print("Epoch:", epoch)
        print(report['samples avg'])


def calculate_metrics(pred, target, threshold=0.5):
    pred = np.array(pred > threshold, dtype=float)
    return classification_report(target, pred, zero_division=1, output_dict=True)
