import pygame
import numpy as np

class NeuralNetworkVisualizer:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        
        # Colors
        self.bg_color = (40, 40, 40)
        self.node_color_low = (50, 50, 200)  # Blue for low activation
        self.node_color_high = (200, 50, 50)  # Red for high activation
        self.positive_weight_color = (50, 50, 200)  # Blue
        self.negative_weight_color = (200, 50, 50)  # Red
        self.text_color = (200, 200, 200)
        
        # Sizes
        self.node_radius = 15
        self.max_weight_thickness = 4
        
        # Spacing
        self.h_margin = 40
        self.v_margin = 30
        
        # Font for labels
        self.font = pygame.font.SysFont(None, 18)
        
        # For animation
        self.target_weights = []
        self.current_weights = []
        self.target_activations = []
        self.current_activations = []
        
        # Input/output labels
        self.input_labels = ["Distance", "Angle", "Energy"]
        self.output_labels = ["Turn Left", "Turn Right", "Forward"]
    
    def update(self, network, inputs=None, outputs=None):
        """Update the visualization with new network data"""
        # Update activations from network
        self.target_activations = []
        for i, layer_activations in enumerate(network.activations):
            if i == 0 and inputs is not None:
                # Use provided inputs for input layer if available
                self.target_activations.append(np.array(inputs))
            elif i == len(network.activations) - 1 and outputs is not None:
                # Use provided outputs for output layer if available
                self.target_activations.append(np.array(outputs))
            else:
                self.target_activations.append(layer_activations)
        
        # Initialize current activations if not already set
        if not self.current_activations:
            self.current_activations = [np.zeros_like(a) for a in self.target_activations]
            
        # Smooth transition for activations (70% towards target)
        for i in range(len(self.current_activations)):
            self.current_activations[i] = self.current_activations[i] * 0.3 + self.target_activations[i] * 0.7
        
        # Update weights from network
        self.target_weights = []
        for i in range(len(network.weights)):
            self.target_weights.append(network.weights[i])
            
        # Initialize current weights if not already set
        if not self.current_weights:
            self.current_weights = [np.zeros_like(w) for w in self.target_weights]
            
        # Smooth transition for weights (70% towards target)
        for i in range(len(self.current_weights)):
            self.current_weights[i] = self.current_weights[i] * 0.3 + self.target_weights[i] * 0.7
    
    def draw(self, screen):
        """Draw the neural network visualization"""
        # Draw background
        pygame.draw.rect(screen, self.bg_color, 
                        (self.x, self.y, self.width, self.height))
        
        # Draw title
        title_font = pygame.font.SysFont(None, 24)
        title = title_font.render("Neural Network (Best Agent)", True, self.text_color)
        screen.blit(title, (self.x + (self.width - title.get_width()) // 2, self.y + 5))
        
        # Check if we have valid data
        if not self.current_activations or not self.current_weights:
            return
        
        # Calculate positions for each node
        layer_positions = []
        max_nodes = max(len(a) for a in self.current_activations)
        layer_count = len(self.current_activations)
        
        # Calculate horizontal spacing
        h_spacing = (self.width - 2 * self.h_margin) / (layer_count - 1) if layer_count > 1 else 0
        
        for i, layer in enumerate(self.current_activations):
            nodes = []
            layer_size = len(layer)
            
            # Calculate vertical spacing for this layer
            v_spacing = (self.height - 2 * self.v_margin) / (layer_size - 1) if layer_size > 1 else 0
            
            for j in range(layer_size):
                x = self.x + self.h_margin + i * h_spacing
                y = self.y + self.v_margin + j * v_spacing
                if layer_size == 1:
                    # Center single nodes vertically
                    y = self.y + self.height // 2
                nodes.append((x, y))
            
            layer_positions.append(nodes)
        
        # Draw connections (weights) between layers
        for i in range(len(self.current_weights)):
            for j in range(self.current_activations[i].size):
                for k in range(self.current_activations[i+1].size):
                    weight = self.current_weights[i][j, k]
                    
                    # Determine color based on weight sign
                    if weight > 0:
                        color = self.positive_weight_color
                    else:
                        color = self.negative_weight_color
                    
                    # Determine thickness based on weight magnitude
                    thickness = min(self.max_weight_thickness, max(1, abs(int(weight * 3))))
                    
                    # Draw the connection
                    start_pos = layer_positions[i][j]
                    end_pos = layer_positions[i+1][k]
                    pygame.draw.line(screen, color, start_pos, end_pos, thickness)
        
        # Draw nodes
        for i, layer in enumerate(self.current_activations):
            for j, activation in enumerate(layer):
                # Map activation (0-1) to color (blue-red)
                color_r = min(255, max(0, int(self.node_color_low[0] * (1 - activation) + self.node_color_high[0] * activation)))
                color_g = min(255, max(0, int(self.node_color_low[1] * (1 - activation) + self.node_color_high[1] * activation)))
                color_b = min(255, max(0, int(self.node_color_low[2] * (1 - activation) + self.node_color_high[2] * activation)))
                color = (color_r, color_g, color_b)
                
                # Draw the node
                pos = layer_positions[i][j]
                pygame.draw.circle(screen, color, pos, self.node_radius)
                
                # Draw node outline
                pygame.draw.circle(screen, (120, 120, 120), pos, self.node_radius, 1)
                
                # Draw activation value text
                text = self.font.render(f"{activation:.2f}", True, (240, 240, 240))
                text_rect = text.get_rect(center=pos)
                screen.blit(text, text_rect)
                
                # Add labels for input and output layers
                if i == 0:  # Input layer
                    if j < len(self.input_labels):
                        label = self.font.render(self.input_labels[j], True, self.text_color)
                        screen.blit(label, (pos[0] - label.get_width() - 5, pos[1] - 8))
                elif i == len(self.current_activations) - 1:  # Output layer
                    if j < len(self.output_labels):
                        label = self.font.render(self.output_labels[j], True, self.text_color)
                        screen.blit(label, (pos[0] + self.node_radius + 5, pos[1] - 8))