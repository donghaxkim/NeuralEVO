import pygame
import math
import numpy as np
import random
from neural_network import NeuralNetwork

class Agent:
    def __init__(self, x, y, environment, brain=None):
        # Position and movement
        self.position_x = x
        self.position_y = y
        self.direction = random.uniform(0, 2 * math.pi)
        self.speed = 100
        self.turn_rate = 3.0
        
        # Agent properties
        self.radius = 10
        self.alive = True
        self.energy = 100
        self.food_eaten = 0
        self.color = (0, 200, 0)
        self.direction_indicator_color = (200, 0, 0)
        
        # Vision properties
        self.vision_radius = 120
        self.vision_angle = math.pi
        self.vision_color = (200, 200, 200, 50)  #semi transparent
        
        # Environment
        self.environment = environment
        
        # Neural network
        if brain is not None:
            self.brain = brain.copy()
        else:
            # NN architecture:
            # 3 inputs: distance to food, angle to food, energy level
            # 3 outputs: turn left, turn right, move forward
            self.brain = NeuralNetwork([3, 8, 3])
        
        # For visualization
        self.last_inputs = [0, 0, 0]
        self.last_outputs = [0, 0, 0]
        self.target_food = None
        
        # To track if agent is stuck
        self.last_positions = []
        self.stuck_counter = 0
        self.is_stuck = False
    
    def update(self, foods, dt):
        if not self.alive:
            return
        
        # Lose energy over time
        self.energy -= 0.1 * dt * 60
        
        if self.energy <= 0:
            self.alive = False
            return
        
        # Find closest food in vision cone
        closest_food = None
        closest_distance = float('inf')
        closest_angle = 0
        
        for food in foods:
            # Calculate distance to food
            dx = food.position_x - self.position_x
            dy = food.position_y - self.position_y
            distance = math.sqrt(dx*dx + dy*dy)
            
            # Skip if food is too far
            if distance > self.vision_radius:
                continue
            
            # Calculate angle to food relative to agent's direction
            food_angle = math.atan2(dy, dx)
            angle_diff = self.normalize_angle(food_angle - self.direction)
            
            # Check if food is within vision cone
            if abs(angle_diff) <= self.vision_angle / 2:
                if distance < closest_distance:
                    closest_distance = distance
                    closest_food = food
                    closest_angle = angle_diff
        
        self.target_food = closest_food
        
        # Prepare neural network inputs
        if closest_food is not None:
            # Normalize inputs
            normalized_distance = closest_distance / self.vision_radius  # 0 to 1
            normalized_angle = closest_angle / (self.vision_angle / 2)  # -1 to 1
            normalized_energy = self.energy / 100  # 0 to 1
            
            inputs = [normalized_distance, normalized_angle, normalized_energy]
        else:
            # No food in sight
            inputs = [1.0, 0.0, self.energy / 100]
        
        self.last_inputs = inputs
        
        # Get neural network decision
        outputs = self.brain.forward(inputs).copy()
        self.last_outputs = outputs.copy()
        
        # Check if agent is stuck
        self.last_positions.append((self.position_x, self.position_y))
        if len(self.last_positions) > 20:
            self.last_positions.pop(0)
            
            # Check if position hasn't changed significantly
            if len(self.last_positions) >= 20:
                x_positions = [p[0] for p in self.last_positions]
                y_positions = [p[1] for p in self.last_positions]
                x_diff = max(x_positions) - min(x_positions)
                y_diff = max(y_positions) - min(y_positions)
                
                if x_diff < 10 and y_diff < 10:
                    self.stuck_counter += 1
                    if self.stuck_counter > 5:
                        self.is_stuck = True
                else:
                    self.stuck_counter = 0
                    self.is_stuck = False
        
        # Apply stronger food pull if agent is stuck or food is visible
        food_pull_modifier = 1.0
        if self.is_stuck:
            food_pull_modifier = 3.0 # Stronger pull when stuck

            if random.random() < 0.1:
                self.direction += random.uniform(-math.pi/2, math.pi/2)
        
        # Apply pull toward food if visible - STRONGER, MORE NOTICEABLE PULL
        if closest_food is not None:
            # Calculate pull strength - stronger when closer to food
            pull_strength = 0.5 * (1.0 - normalized_distance) * food_pull_modifier
            
            # Modify outputs based on the angle to food - MUCH MORE SIGNIFICANT NOW
            if closest_angle < 0:  # Food is to the left
                # Increase "turn left" output significantly
                outputs[0] += pull_strength
            else:
                
                outputs[1] += pull_strength
                
            # Always increase "move forward" to encourage approach
            outputs[2] += pull_strength * 0.7
        
        # Determine action based on highest output
        action = np.argmax(outputs)
        
        # Always be moving
        if random.random() < 0.05:
            action = 2
        
        # Execute action
        if action == 0:
            self.direction -= self.turn_rate * dt
        elif action == 1:
            self.direction += self.turn_rate * dt
        elif action == 2:
            # Calculate new position
            move_distance = self.speed * dt
            new_x = self.position_x + math.cos(self.direction) * move_distance
            new_y = self.position_y + math.sin(self.direction) * move_distance
            
            # Handle boundary collisions more intelligently
            # Bounce off walls by reversing direction component
            if new_x < self.radius:
                new_x = self.radius + 1
                self.direction = math.pi - self.direction  # Horizontal bounce
            elif new_x > self.environment.width - self.radius:
                new_x = self.environment.width - self.radius - 1
                self.direction = math.pi - self.direction  # Horizontal bounce
                
            if new_y < self.radius:
                new_y = self.radius + 1
                self.direction = -self.direction  # Vertical bounce
            elif new_y > self.environment.height - self.radius:
                new_y = self.environment.height - self.radius - 1
                self.direction = -self.direction  # Vertical bounce
                
            # Apply direct pull toward food if visible
            if closest_food is not None:
                # Calculate pull vector (normalized and scaled by pull strength)
                pull_strength = 0.25 * (1.0 - normalized_distance) * food_pull_modifier
                pull_x = pull_strength * (closest_food.position_x - self.position_x)
                pull_y = pull_strength * (closest_food.position_y - self.position_y)
                
                # Add pull to movement
                new_x += pull_x
                new_y += pull_y
            
            # Update position
            self.position_x = new_x
            self.position_y = new_y
            
        # Normalize the direction to keep it within [0, 2π]
        self.direction = self.direction % (2 * math.pi)
    
    def check_food_collision(self, food):
        if not self.alive:
            return False
            
        dx = food.position_x - self.position_x
        dy = food.position_y - self.position_y
        distance = math.sqrt(dx*dx + dy*dy)
        
        return distance < self.radius + food.radius
    
    def draw(self, screen):
        if not self.alive:
            return
            
        # Draw vision cone (semi-transparent)
        vision_surface = pygame.Surface((self.vision_radius * 2, self.vision_radius * 2), pygame.SRCALPHA)
        center = (self.vision_radius, self.vision_radius)
        
        # Calculate start and end angles for arc
        start_angle = self.direction - self.vision_angle / 2
        end_angle = self.direction + self.vision_angle / 2
        
        # Draw vision cone as a pie wedge
        pygame.draw.arc(vision_surface, self.vision_color, 
                         (0, 0, self.vision_radius * 2, self.vision_radius * 2),
                         start_angle, end_angle, self.vision_radius)
        
        # Create points for the wedge
        points = [center]
        for angle in np.linspace(start_angle, end_angle, 20):
            x = center[0] + math.cos(angle) * self.vision_radius
            y = center[1] + math.sin(angle) * self.vision_radius
            points.append((x, y))
        points.append(center)
        
        # Draw the wedge
        if len(points) > 2:
            pygame.draw.polygon(vision_surface, self.vision_color, points)
        
        # Display the vision surface
        screen.blit(vision_surface, 
                   (self.position_x - self.vision_radius, 
                    self.position_y - self.vision_radius))
        
        # Draw line to target food if visible
        if self.target_food is not None:
            # Get normalized distance for visual effects
            dx = self.target_food.position_x - self.position_x
            dy = self.target_food.position_y - self.position_y
            distance = math.sqrt(dx*dx + dy*dy)
            normalized_distance = min(1.0, distance / self.vision_radius)
            
            # Draw line to food - brighter when closer
            line_color = (
                min(255, int(200 + (1 - normalized_distance) * 55)),
                min(255, int(200 + (1 - normalized_distance) * 55)),
                0
            )
            pygame.draw.line(screen, line_color, 
                            (self.position_x, self.position_y),
                            (self.target_food.position_x, self.target_food.position_y), 1)
            
            # Draw midpoint dot, larger when closer to food
            mid_x = (self.position_x + self.target_food.position_x) / 2
            mid_y = (self.position_y + self.target_food.position_y) / 2
            dot_size = int(3 + (1 - normalized_distance) * 3)
            
            # Add subtle glow effect for the dot
            for i in range(3):
                glow_size = dot_size + i*2
                glow_alpha = int(150 - i*50)
                glow_surface = pygame.Surface((glow_size*2, glow_size*2), pygame.SRCALPHA)
                glow_color = (255, 255, 0, glow_alpha)
                pygame.draw.circle(glow_surface, glow_color, (glow_size, glow_size), glow_size)
                screen.blit(glow_surface, (int(mid_x)-glow_size, int(mid_y)-glow_size))
                
            pygame.draw.circle(screen, (255, 255, 0), (int(mid_x), int(mid_y)), dot_size)
        
        # Draw agent body
        pygame.draw.circle(screen, self.color, (int(self.position_x), int(self.position_y)), self.radius)
        
        # Draw direction indicator (line pointing in direction of movement)
        end_x = self.position_x + math.cos(self.direction) * self.radius
        end_y = self.position_y + math.sin(self.direction) * self.radius
        pygame.draw.line(screen, self.direction_indicator_color, 
                        (self.position_x, self.position_y), 
                        (end_x, end_y), 2)
    
    def get_fitness(self):
        return 10 * self.food_eaten
    
    @staticmethod
    def normalize_angle(angle):
        """Normalize angle to be between -pi and pi"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle