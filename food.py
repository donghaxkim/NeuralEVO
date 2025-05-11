import pygame

class Food:
    def __init__(self, x, y):
        self.position_x = x
        self.position_y = y
        self.radius = 5
        self.color = (200, 30, 30)
        self.glow_color = (200, 50, 50, 100)
    
    def draw(self, screen):
        
        glow_surface = pygame.Surface((self.radius * 4, self.radius * 4), pygame.SRCALPHA)
        pygame.draw.circle(glow_surface, self.glow_color, (self.radius * 2, self.radius * 2), self.radius * 2)
        screen.blit(glow_surface, 
                   (int(self.position_x - self.radius * 2), 
                    int(self.position_y - self.radius * 2)))
        
        # Draw food
        pygame.draw.circle(screen, self.color, 
                          (int(self.position_x), int(self.position_y)), 
                          self.radius)