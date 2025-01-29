# ESP For MNIST digits recognition

### Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from deap import base, creator, tools
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

### Define the model
class FixedNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FixedNeuralNetwork, self).__init__()
        self.input_to_hidden = nn.Linear(input_size, hidden_size)
        self.hidden_to_output = nn.Linear(hidden_size, output_size)
        
        for param in self.input_to_hidden.parameters():
            param.requires_grad = False
        for param in self.hidden_to_output.parameters():
            param.requires_grad = False        
    
    def forward(self, x):
        x = F.relu(self.input_to_hidden(x))
        x = self.hidden_to_output(x)
        return x

# Initialize the network
input_size = 28 * 28  # MNIST input size
hidden_size = 16      # Hidden layer size
output_size = 10      # Output size (digits 0-9)

network = FixedNeuralNetwork(input_size, hidden_size, output_size)
### Load Dataset
# Flatten and normalize images from (28*28) to (784)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
    transforms.Lambda(lambda x: x.view(-1))  # Flatten the images
])

# Load training and test data
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Split training data into training and validation sets
train_size = int(0.9 * len(train_data))  # 90% for training
val_size = len(train_data) - train_size  # Remaining 10% for validation
train_data, val_data = random_split(train_data, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)  # No shuffling for validation
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
### Initalize sub populations
population_size = 20  # Number of candidates per sub-population
sub_populations = [
    np.random.uniform(-1, 1, (population_size, input_size + 1))  # +1 for bias
    for _ in range(hidden_size)
]

### Create Fitness Function
def evaluate_fitness(individual, neuron_idx, network, data_loader):
    with torch.no_grad():
        # Assign weights and bias to the specific neuron
        network.input_to_hidden.weight[neuron_idx, :] = torch.tensor(individual[:-1])
        network.input_to_hidden.bias[neuron_idx] = torch.tensor(individual[-1])

        # Test the network and calculate loss
        total_loss = 0
        i = 0
        for inputs, labels in data_loader:
            inputs = inputs.view(inputs.size(0), -1)  # Flatten input
            outputs = network(inputs)
            loss = F.cross_entropy(outputs, labels)
            total_loss += loss.item()
            
            i += 1
            if i == 96:
                break
        
    return -total_loss,  # Negative loss as fitness

# Define fitness as a single objective to maximize (accuracy or negative loss)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

# Define an individual (candidate weights for one neuron)
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.uniform, -1, 1)  # Weights in [-1, 1]
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=input_size + 1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate_fitness)
toolbox.register("mate", tools.cxBlend, alpha=0.5)  # Blend crossover
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)  # Gaussian mutation
toolbox.register("select", tools.selTournament, tournsize=3)  # Tournament selection

def evolve_sub_population(neuron_idx, sub_population, generations, network, data_loader):
    # Initialize population
    population = toolbox.population(n=population_size)

    # Assign sub-population weights to individuals
    for i, individual in enumerate(population):
        individual[:] = sub_population[i]

    # Evolution process
    for gen in range(generations):
        # Evaluate fitness for each individual
        fitnesses = [toolbox.evaluate(ind, neuron_idx, network, data_loader) for ind in population]
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # Select, mate, and mutate to create the next generation
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if np.random.rand() < 0.7:  # Crossover probability
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if np.random.rand() < 0.2:  # Mutation probability
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Replace old population with new offspring
        population[:] = offspring

    # Return the best individual (weights for the neuron)
    best_ind = tools.selBest(population, 1)[0]
    return best_ind

### RUN ESP
generations = 10
criterion = nn.CrossEntropyLoss()

for neuron_idx in range(hidden_size):
    print(f"{neuron_idx + 1}/{hidden_size}...")
    best_weights = evolve_sub_population(neuron_idx, sub_populations[neuron_idx], generations, network, train_loader)

    # Assign best weights back to the sub-population
    sub_populations[neuron_idx] = best_weights
    network.input_to_hidden.weight[neuron_idx, :] = torch.tensor(best_weights[:-1])
    network.input_to_hidden.bias[neuron_idx] = torch.tensor(best_weights[-1])
    
    # Calculate Loss 
    val_loss = 0.0
    with torch.no_grad():  # Disable gradient computation for validation
        for inputs, labels in val_loader:
            # Forward pass (no backward pass)
            outputs = network(inputs)
            # Compute the validation loss
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    # Calculate average validation loss for the epoch
    avg_val_loss = val_loss / len(val_loader)

    # Print the losses for this epoch
    print(f"Loss: {avg_val_loss:.4f}")
    
### Test ESP
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.view(inputs.size(0), -1)
        outputs = network(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Final Accuracy: {100 * correct / total:.2f}%")