import torch
import torch.nn as nn
import torch.optim as optim

class FeedForwardNN(nn.Module): 
    """
    Simple feedforward neural network 
    
    784 input neurons;
    128 neurons in first hidden layer;
    64 neurons in second hidden layer;
    ReLu activation function
    """
    def __init__(self):
        super(FeedForwardNN, self).__init__()
        
        # Define the layers
        self.fc1 = nn.Linear(784, 128)  # Input to first hidden layer
        self.fc2 = nn.Linear(128, 64)   # First to second hidden layer
        self.fc3 = nn.Linear(64, 10)    # Second hidden layer to output
        
        self.relu = nn.ReLU()  # Activation function
        
    def forward(self, x):
        x = self.relu(self.fc1(x))  # First hidden layer
        x = self.relu(self.fc2(x))  # Second hidden layer
        x = self.fc3(x)             # Output layer
        return x
    
    def optimizer(self):
        return optim.Adam(self.parameters(), lr=0.001)
    
    def loss(self):
        return nn.CrossEntropyLoss() 

    def train_model(self, train_loader, val_loader, num_epochs=10):
        optimizer = self.optimizer()
        criterion = self.loss()
        self.train()  # Set the model to training mode

        for epoch in range(num_epochs):
            running_loss = 0.0
            # Training loop
            for inputs, labels in train_loader:
                optimizer.zero_grad()  # Zero the gradients
                
                # Forward pass
                outputs = self(inputs)
                
                # Compute the loss
                loss = criterion(outputs, labels)
                
                # Backward pass 
                loss.backward()
                
                # Optimize the model
                optimizer.step()
                
                running_loss += loss.item()

            # Calculate average training loss for the epoch
            avg_train_loss = running_loss / len(train_loader)

            # Now compute validation loss
            self.eval()  # Set the model to evaluation mode
            val_loss = 0.0
            with torch.no_grad():  # Disable gradient computation for validation
                for inputs, labels in val_loader:
                    # Forward pass (no backward pass)
                    outputs = self(inputs)
                    # Compute the validation loss
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
            
            # Calculate average validation loss for the epoch
            avg_val_loss = val_loss / len(val_loader)

            # Print the losses for this epoch
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                f"Training Loss: {avg_train_loss:.4f}, "
                f"Validation Loss: {avg_val_loss:.4f}")

            self.train()  # Set the model back to training mode after validation
    
    def evaluate_model(self, test_loader):
        self.eval()  # Set the model to evaluation mode
        correct = 0
        total = 0
        with torch.no_grad():  # No gradients needed for evaluation
            for inputs, labels in test_loader:
                outputs = self(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = (correct / total) * 100
        print(f'Test Accuracy: {accuracy:.2f}%')
