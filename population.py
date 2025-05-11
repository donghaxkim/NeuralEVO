import random
import numpy as np
from agent import Agent

class Population:
    def __init__(self, size, environment):
        self.size = size
        self.environment = environment
        self.agents = []
        self.initialize_population()
    
    def initialize_population(self):
        """Initialize a new population of agents with random positions"""
        self.agents = []
        margin = 50  # Keep agents away from edges at start
        
        for _ in range(self.size):
            x = random.uniform(margin, self.environment.width - margin)
            y = random.uniform(margin, self.environment.height - margin)
            self.agents.append(Agent(x, y, self.environment))
    
    def update(self, foods, dt):
        """Update all agents in the population"""
        for agent in self.agents:
            agent.update(foods, dt)
    
    def get_best_agent(self):
        """Get the agent with the highest fitness"""
        if not self.agents:
            return None
            
        living_agents = [agent for agent in self.agents if agent.alive]
        if not living_agents:
            return self.agents[0]  # Return any agent if none are alive
            
        return max(living_agents, key=lambda agent: agent.get_fitness())
    
    def evolve(self):
        """Evolve the population for the next generation"""
        # Calculate fitness for all agents
        fitnesses = [agent.get_fitness() for agent in self.agents]
        
        # Check if any agent has fitness (avoid division by zero)
        if sum(fitnesses) == 0:
            # If all agents have zero fitness, reinitialize population
            self.initialize_population()
            return
        
        # Create new population
        new_agents = []
        
        # Keep the best agent (elitism)
        best_agent = self.agents[np.argmax(fitnesses)]
        new_agents.append(Agent(
            random.uniform(50, self.environment.width - 50),
            random.uniform(50, self.environment.height - 50),
            self.environment,
            best_agent.brain
        ))
        
        # Selection probability proportional to fitness
        selection_probs = np.array(fitnesses) / sum(fitnesses)
        
        # Create rest of the new population
        for _ in range(self.size - 1):
            # Select parents
            parent1_idx = np.random.choice(len(self.agents), p=selection_probs)
            parent2_idx = np.random.choice(len(self.agents), p=selection_probs)
            
            parent1 = self.agents[parent1_idx]
            parent2 = self.agents[parent2_idx]
            
            # Crossover
            if random.random() < 0.7:  # 70% chance of crossover
                child_brain = parent1.brain.crossover(parent2.brain)
            else:
                # No crossover, just copy the better parent
                if fitnesses[parent1_idx] > fitnesses[parent2_idx]:
                    child_brain = parent1.brain.copy()
                else:
                    child_brain = parent2.brain.copy()
            
            # Mutation
            child_brain.mutate(mutation_rate=0.1, mutation_scale=0.2)
            
            # Create new agent with evolved brain
            x = random.uniform(50, self.environment.width - 50)
            y = random.uniform(50, self.environment.height - 50)
            new_agents.append(Agent(x, y, self.environment, child_brain))
        
        # Replace old population with new one
        self.agents = new_agents