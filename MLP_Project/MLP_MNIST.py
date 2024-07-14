import torch.nn as nn
import torch.optim as optim
import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Fix a random seed to get reproducible results
torch.manual_seed(1)

# Prepare the training set with normalization
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
training_data = datasets.MNIST('/data', train=True, download=True, transform=transform)
validation_data = datasets.MNIST('/data', train=False, transform=transform)

batchsize = 64

# Train dataloader
train_loader = DataLoader(training_data, batch_size=batchsize, shuffle=True)
validation_loader = DataLoader(validation_data, batch_size=batchsize, shuffle=False)

# Define the model
class MLP(nn.Module):
    def __init__(self, num_inputs, num_hidden_layer_nodes, num_outputs):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(num_inputs, num_hidden_layer_nodes),
            nn.ReLU(),
            nn.Dropout(0.2),  # Add Dropout layer
            nn.Linear(num_hidden_layer_nodes, num_hidden_layer_nodes),
            nn.ReLU(),
            nn.Dropout(0.2),  # Add Dropout layer
            nn.Linear(num_hidden_layer_nodes, num_outputs)
        )
        
    def forward(self, x):
        return self.model(x)

num_inputs = 28*28
num_hidden_layer_nodes = 512
num_outputs = 10

# Check if CUDA is available and move model to GPU if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Training on {device}")


def train():
    model.train()
    running_loss = 0
    running_correct = 0

    for (x_train, y_train) in train_loader:
        x_train, y_train = x_train.to(device), y_train.to(device)  # Move data to device
        
        # Forward pass
        x_train = x_train.view(x_train.shape[0], -1)
        y = model(x_train)
      
        # Compute loss
        loss = loss_function(y, y_train)
        running_loss += loss.item()

        # Compute accuracy
        y_pred = y.argmax(dim=1)
        correct = torch.sum(y_pred == y_train)
        running_correct += correct

        # Zero gradients, perform a backward pass, and update the weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
  
    return running_loss / len(train_loader), running_correct.item() / len(train_loader.dataset)

def validate():
    model.eval()
    running_loss = 0
    running_correct = 0
    with torch.no_grad():
        for (x_val, y_val) in validation_loader:
            x_val, y_val = x_val.to(device), y_val.to(device)  # Move data to device
            
            # Forward pass
            x_val = x_val.view(x_val.shape[0], -1)
            y = model(x_val)

            # Compute accuracy
            y_pred = y.argmax(dim=1)
            correct = torch.sum(y_pred == y_val)
            running_correct += correct

            # Compute loss
            loss = loss_function(y, y_val)
            running_loss += loss.item()
      
    return running_loss / len(validation_loader), running_correct.item() / len(validation_loader.dataset)

# Training parameters
num_epochs = 20 

# Construct model
model = MLP(num_inputs, num_hidden_layer_nodes, num_outputs).to(device)

# Define loss function
loss_function = nn.CrossEntropyLoss()

# Define optimizer
optimizer = optim.SGD(model.parameters(), lr=1e-2)

train_loss_history = []
val_loss_history = []
train_acc_history = []
val_acc_history = []

print("Starting Training...")
for ep in range(num_epochs):
    train_loss, train_acc = train()
    val_loss, val_acc = validate()
    print("Epoch: {}, Train Loss = {:.3f}, Train Acc = {:.3f}, Val Loss = {:.3f}, Val Acc = {:.3f}".
          format(ep, train_loss, train_acc, val_loss, val_acc))
    train_loss_history.append(train_loss)
    val_loss_history.append(val_loss)
    train_acc_history.append(train_acc)
    val_acc_history.append(val_acc)

# Save the model and its parameters
model_file = 'mnist_mlp.pt'
torch.save({
    'epoch': num_epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss_history': train_loss_history,
    'val_loss_history': val_loss_history,
    'train_acc_history': train_acc_history,
    'val_acc_history': val_acc_history,
}, model_file)
print(f"Model and parameters saved to {model_file}")

# Plot the loss and accuracy
import matplotlib.pyplot as plt

plt.figure()
plt.plot(train_loss_history, label='Train Loss')
plt.plot(val_loss_history, label='Val Loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss')
plt.show()

plt.figure()
plt.plot(train_acc_history, label='Train Accuracy')
plt.plot(val_acc_history, label='Val Accuracy')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy')
plt.show()

