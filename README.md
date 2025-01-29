# MNIST characters recognition through neuro evolutionary algorithms.
This repository contains all the necessary materials for the Bio-Inspired Artificial Intelligence course at the University of Trento for the academic year 2024/2025.

Handwritten digit recognition is a challenging problem that traditional deterministic algorithms struggle to solve. In this project, we explore the use of neuro-evolution to address this task. Specifically, we focus on recognizing handwritten digits from the MNIST dataset. The goal is to classify a grayscale 28x28 pixel image into the correct digit.

The project follows a three-step approach:

1. **Backpropagation**: We'll first solve the problem using traditional backpropagation with a neural network.
2. **Fixed-size Neural Network**: Next, we use an evolutionary algorithm to evolve the weights and biases of a fixed-size neural network.
3. **NEAT Algorithm**: Finally, we employ the NEAT (NeuroEvolution of Augmenting Topologies) algorithm, which dynamically evolves both the architecture (shape) and the weights of the neural network.

> **Important Note:**  
> If you want to understand how the code works, check the Jupyter notebooks version, not the Python version.


### Requirements
```
pip install neat-python
pip install neat-python[visualize]
pip install torch torchvision
```
### Reproducing the paper's results:
#### Backpropagation
execute `python3 src/backpropagation/main.py`

#### NEAT
execute `python3 src/neat/main.py`

#### Fixed topology
execute `python3 src/fixed_topology/main.py`

