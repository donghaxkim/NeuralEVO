import pygame
import sys
import random
import math
import numpy as np
from agent import Agent
from food import Food
from environment import Environment
from population import Population
from neural_network_visualizer import NeuralNetworkVisualizer

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
SIMULATION_WIDTH = 900
SIMULATION_HEIGHT = 800
INFO_WIDTH = 300
INFO_HEIGHT = 800
BACKGROUND_COLOR = (30, 30, 30)
TEXT_COLOR = (200, 200, 200)

# Initialize screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Neural Network Evolution Simulation")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)

# Create environment
environment = Environment(SIMULATION_WIDTH, SIMULATION_HEIGHT)

# Initialize population
population_size = 50
population = Population(population_size, environment)

# Create food
food_count = 20
foods = []
for _ in range(food_count):
    # Ensure food isn't placed too close to the edges
    x = random.randint(30, SIMULATION_WIDTH-30)
    y = random.randint(30, SIMULATION_HEIGHT-30)
    foods.append(Food(x, y))

# Create neural network visualizer
nn_visualizer = NeuralNetworkVisualizer(
    SIMULATION_WIDTH + 20, 
    20, 
    INFO_WIDTH - 40, 
    300
)

# Game state
paused = False
generation = 1
running = True

# Stats
best_fitness = 0
generation_time = 0
generation_timeout = 45  # seconds before forcing next generation

def draw_info_panel():
    # Draw background for info panel
    pygame.draw.rect(screen, (40, 40, 40), 
                    (SIMULATION_WIDTH, 0, INFO_WIDTH, INFO_HEIGHT))
    
    # Draw divider line
    pygame.draw.line(screen, (100, 100, 100), 
                    (SIMULATION_WIDTH, 0), 
                    (SIMULATION_WIDTH, SCREEN_HEIGHT), 3)
    
    # Draw stats
    alive_count = sum(1 for agent in population.agents if agent.alive)
    stuck_count = sum(1 for agent in population.agents if agent.alive and agent.is_stuck)
    
    stats = [
        f"Generation: {generation}",
        f"Best Fitness: {best_fitness:.1f}",
        f"Agents Alive: {alive_count}/{population_size}",
        f"Stuck Agents: {stuck_count}",
        f"Time: {generation_time:.1f}s",
        f"FPS: {int(clock.get_fps())}",
        "",
        "Controls:",
        "SPACE - Pause/Resume",
        "R - Reset simulation",
        "N - Next generation",
        "Q - Quit"
    ]
    
    y_offset = 320
    for stat in stats:
        text = font.render(stat, True, TEXT_COLOR)
        screen.blit(text, (SIMULATION_WIDTH + 20, y_offset))
        y_offset += 30

def reset_food():
    """Reset all food to new random positions"""
    for food in foods:
        # Place food away from walls
        food.position_x = random.randint(30, SIMULATION_WIDTH-30)
        food.position_y = random.randint(30, SIMULATION_HEIGHT-30)

def reset_simulation():
    global generation, best_fitness, generation_time
    generation = 1
    best_fitness = 0
    generation_time = 0
    population.initialize_population()
    reset_food()

def next_generation():
    global generation, generation_time
    generation += 1
    generation_time = 0
    
    # Evolve population
    population.evolve()
    reset_food()

# Main game loop
while running:
    dt = clock.tick(60) / 1000.0  # Delta time in seconds
    
    if not paused:
        generation_time += dt
    
    # Process events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False
            elif event.key == pygame.K_SPACE:
                paused = not paused
            elif event.key == pygame.K_r:
                reset_simulation()
            elif event.key == pygame.K_n:
                next_generation()
    
    # Clear screen
    screen.fill(BACKGROUND_COLOR)
    
    # Update and draw all objects if not paused
    if not paused:
        # Update agents
        population.update(foods, dt)
        
        # Check if agent eats food
        for agent in population.agents:
            if agent.alive:
                for food in foods:
                    if agent.check_food_collision(food):
                        # Agent eats food
                        agent.energy += 50
                        agent.food_eaten += 1
                        # Reset food position (away from walls)
                        food.position_x = random.randint(30, SIMULATION_WIDTH-30)
                        food.position_y = random.randint(30, SIMULATION_HEIGHT-30)
        
        # Update best fitness
        current_best = max([agent.get_fitness() for agent in population.agents])
        best_fitness = max(best_fitness, current_best)
        
        # Check if generation should end
        all_dead = all(not agent.alive for agent in population.agents)
        timeout = generation_time > generation_timeout
        
        if all_dead or timeout:
            next_generation()
    
    # Draw environment
    environment.draw(screen)
    
    # Draw food
    for food in foods:
        food.draw(screen)
    
    # Draw agents
    for agent in population.agents:
        if agent.alive:
            agent.draw(screen)
    
    # Get best agent for neural network visualization
    best_agent = population.get_best_agent()
    if best_agent:
        nn_visualizer.update(best_agent.brain, best_agent.last_inputs, best_agent.last_outputs)
        nn_visualizer.draw(screen)
    
    # Draw info panel
    draw_info_panel()
    
    # Update display
    pygame.display.flip()

# Quit Pygame
pygame.quit()
sys.exit()