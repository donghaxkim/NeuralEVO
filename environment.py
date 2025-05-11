import pygame

class Environment:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.border_color = (80, 80, 80)
        self.border_width = 2
    
    def draw(self, screen):
        # Draw border around the environment
        pygame.draw.rect(screen, self.border_color, 
                        (0, 0, self.width, self.height), 
                        self.border_width)