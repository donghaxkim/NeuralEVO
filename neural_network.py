import numpy as np
import copy

class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        for i in range(1, self.num_layers):
            # Initialize weights with small random values
            w = np.random.randn(self.layer_sizes[i-1], self.layer_sizes[i]) * 0.1
            self.weights.append(w)
            
            # Initialize biases with zeros
            b = np.zeros((1, self.layer_sizes[i]))
            self.biases.append(b)
        
        # For visualization
        self.activations = [np.zeros(size) for size in layer_sizes]
    
    def forward(self, inputs):
        """
        Forward pass through the neural network
        """
        # Convert inputs to numpy array
        a = np.array(inputs).reshape(1, -1)
        self.activations[0] = a.flatten()
        
        for i in range(self.num_layers - 1):
            # Calculate z = a*w + b
            z = np.dot(a, self.weights[i]) + self.biases[i]
            
            # Apply sigmoid activation function
            a = self.sigmoid(z)
            self.activations[i+1] = a.flatten()
        
        return a.flatten()
    
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))
    
    def copy(self):
        """
        Create a deep copy of the neural network
        """
        return copy.deepcopy(self)
    
    def mutate(self, mutation_rate=0.1, mutation_scale=0.2):
        """
        Randomly mutate the weights and biases
        """
        for i in range(len(self.weights)):
            # Mutate weights
            mutation_mask = np.random.random(self.weights[i].shape) < mutation_rate
            mutation = np.random.randn(*self.weights[i].shape) * mutation_scale
            self.weights[i] += mutation * mutation_mask
            
            # Mutate biases
            mutation_mask = np.random.random(self.biases[i].shape) < mutation_rate
            mutation = np.random.randn(*self.biases[i].shape) * mutation_scale
            self.biases[i] += mutation * mutation_mask
    
    def crossover(self, other):
        """
        Perform crossover with another neural network
        """
        if self.layer_sizes != other.layer_sizes:
            raise ValueError("Neural networks must have the same architecture for crossover")
        
        child = self.copy()
        
        for i in range(len(self.weights)):
            # Create crossover mask (50% chance for each parent's weights/biases)
            mask = np.random.random(self.weights[i].shape) < 0.5
            
            # Apply mask for weights
            child.weights[i] = np.where(mask, self.weights[i], other.weights[i])
            
            # Apply mask for biases
            mask = np.random.random(self.biases[i].shape) < 0.5
            child.biases[i] = np.where(mask, self.biases[i], other.biases[i])
        
        return child