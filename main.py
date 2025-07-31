import pygame
import numpy as np
import math
from dqn_agent import DQNAgent

pygame.init()
WIDTH, HEIGHT = 600, 400
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("2D RL Car Simulator with LiDAR and Reward Visualization")

WHITE = (255, 255, 255)
RED = (255, 0, 0)
ROAD_WIDTH = 200
clock = pygame.time.Clock()

# Global constant for car speed to frame rate ratio
CONTROL_RATIO = 0.1667  # pixels per frame per step
FIXED_TIMESTEP = 1.0 / 60.0  # 60 updates per second

class Car:
    def __init__(self):
        self.reset()
        self.last_action = None
        self.speed = 300  # pixels per second (default, will be set in main)

    def reset(self):
        # Set random starting x within the road
        self.x = np.random.randint((WIDTH - ROAD_WIDTH)//2 + 10, (WIDTH + ROAD_WIDTH)//2 - 10)
        self.y = HEIGHT - 50
        self.done = False
        return self.get_state()

    def get_lidar_readings(self):
        # Corrected angles: forward is +90 degrees (flipped)
        angles = [30, 60, 90, 120, 150]
        max_dist = 150
        readings = []
        for angle in angles:
            rad = math.radians(angle)
            dx = math.cos(rad)
            dy = -math.sin(rad)
            hit_type = 0  # 0: nothing, 1: lane marking, 2: road edge
            dist = max_dist
            for d in range(0, max_dist, 1):  # Finer LiDAR sampling for better detection
                px = int(self.x + dx * d)
                py = int(self.y + dy * d)
                if 0 <= px < WIDTH and 0 <= py < HEIGHT:
                    color = win.get_at((px, py))
                    if color == (255, 255, 255, 255):  # Lane marking
                        hit_type = 1
                        dist = d
                        break
                    elif color != (50, 50, 50, 255):  # Road edge (not road or lane marking)
                        hit_type = 2
                        dist = d
                        break
                else:
                    hit_type = 2
                    dist = d
                    break
            # Encode as: normalized_dist, is_lane_marking, is_road_edge
            readings.append([
                dist / max_dist,
                1.0 if hit_type == 1 else 0.0,
                1.0 if hit_type == 2 else 0.0
            ])
        return readings

    def get_state(self):
        # Flatten the lidar readings for the state
        return np.array(self.get_lidar_readings(), dtype=np.float32).flatten()

    def step(self, action, dt=FIXED_TIMESTEP):
        move_dist = self.speed * dt
        if action == 0:
            self.x -= move_dist  # Steer left
        elif action == 1:
            self.x += move_dist  # Steer right
        elif action == 2:
            pass  # No steering

        # Define lane centers
        left_lane_center = WIDTH // 2 - ROAD_WIDTH // 4
        right_lane_center = WIDTH // 2 + ROAD_WIDTH // 4

        # Distance to each lane center
        dist_left = abs(self.x - left_lane_center)
        dist_right = abs(self.x - right_lane_center)

        # Use the smaller distance (closer lane)
        dist_from_lane_center = min(dist_left, dist_right)
        normalized_lane_dist = dist_from_lane_center / (ROAD_WIDTH // 4)

        reward = (1.0 - normalized_lane_dist) * 2

        # Add small reward for not steering
        if action == 2:
            reward += 2

        # Penalize for changing steering action, but not if current action is 'not steering'
        if self.last_action is not None and action != self.last_action and action != 2:
            reward -= 1
        self.last_action = action

        # Instead of car center, use 25% from left and 25% from right for edge checks
        car_width = ROAD_WIDTH // 4
        left_edge_x = self.x - car_width // 4  # 25% from left
        right_edge_x = self.x + car_width // 4  # 25% from right
        road_left = (WIDTH - ROAD_WIDTH) // 2
        road_right = (WIDTH + ROAD_WIDTH) // 2
        if left_edge_x < road_left or right_edge_x > road_right:
            reward = -15
            self.done = True

        return self.get_state(), reward, self.done

    def draw(self):
        # Set car width to 50% of lane width
        car_width = ROAD_WIDTH // 4  # 50% of lane width (ROAD_WIDTH/2 lanes)
        car_height = car_width * 1.5  # Keep height as before or adjust as needed
        pygame.draw.rect(win, RED, (self.x - car_width // 2, self.y - car_height // 2, car_width, car_height))
        # Corrected angles: forward is +90 degrees (flipped)
        angles = [30, 60, 90, 120, 150]
        max_dist = 150
        # Calculate the front-center of the car
        car_front_x = self.x
        car_front_y = self.y - car_height // 2
        # Draw LiDAR range arcs from the front-center
        for angle in angles:
            rad = math.radians(angle)
            end_x = int(car_front_x + math.cos(rad) * max_dist)
            end_y = int(car_front_y - math.sin(rad) * max_dist)
            pygame.draw.line(win, (0, 0, 255), (car_front_x, car_front_y), (end_x, end_y), 1)
        # Draw LiDAR hit points as before, but from the front-center
        for angle in angles:
            rad = math.radians(angle)
            dx = math.cos(rad)
            dy = -math.sin(rad)
            for dist in range(0, max_dist, 1):  # Finer LiDAR sampling for better detection
                px = int(car_front_x + dx * dist)
                py = int(car_front_y + dy * dist)
                if 0 <= px < WIDTH and 0 <= py < HEIGHT:
                    color = win.get_at((px, py))
                    if color != (50, 50, 50, 255):
                        break
                    pygame.draw.circle(win, (0, 255, 0), (px, py), 1)
                else:
                    break

def draw_window(car, reward, scroll):
    if reward < 0:
        bg_color = (255, 200, 200)
    elif reward < 0.5:
        bg_color = (255, 255, 200)
    else:
        bg_color = (200, 255, 200)

    win.fill(bg_color)
    pygame.draw.rect(win, (50, 50, 50), ((WIDTH - ROAD_WIDTH)//2, 0, ROAD_WIDTH, HEIGHT))
    
    # Draw moving dashed center line
    lane_x = WIDTH // 2
    dash_length = 30
    gap_length = 5  # Reduced gap for better LiDAR detection
    y = scroll  # Use positive scroll for correct vertical movement direction
    while y < HEIGHT:
        pygame.draw.rect(
            win,
            (255, 255, 255),
            (lane_x - 2, y, 4, dash_length)
        )
        y += dash_length + gap_length

    car.draw()
    pygame.display.update()

def menu():
    print("Select mode:")
    print("1. Increase speed of all environment elements to match training speed")
    print("2. Reduce training speed to match speed of all environment elements")
    while True:
        choice = input("Enter 1 or 2: ").strip()
        if choice in ("1", "2"):
            return int(choice)
        print("Invalid input. Please enter 1 or 2.")

def main():
    mode = menu()
    car = Car()
    agent = DQNAgent(state_size=15, action_size=3)
    agent.set_lidar_bias()
    # Try to load the latest model if it exists
    import os
    if os.path.exists("dqn_agent_final.pth"):
        print("Loading model from dqn_agent_final.pth...")
        agent.load("dqn_agent_final.pth")
    EPISODES = 500

    dash_length = 30
    gap_length = 20
    scroll = 0
    # Set speed parameters based on menu selection
    if mode == 1:
        car.speed = 600  # pixels per second (fast)
    else:
        car.speed = 200  # pixels per second (slow)
    try:
        for episode in range(EPISODES):
            state = car.reset()
            total_reward = 0
            steps = 0
            scroll = 0 
            time_accum = 0.0
            running = True
            while running:
                dt = clock.tick() / 1000.0  # seconds since last frame
                time_accum += dt
                # Fixed timestep loop
                while time_accum >= FIXED_TIMESTEP:
                    action = agent.choose_action(state)
                    next_state, reward, done = car.step(action, FIXED_TIMESTEP)
                    agent.remember(state, action, reward, next_state, done)
                    agent.replay()
                    state = next_state
                    total_reward += reward
                    steps += 1
                    # Use car.speed for dynamic road speed
                    scroll += car.speed * FIXED_TIMESTEP
                    if scroll >= dash_length + gap_length:
                        scroll = 0
                    if done or steps >= 300:
                        running = False
                        break
                    time_accum -= FIXED_TIMESTEP
                draw_window(car, reward, scroll)
                pygame.event.pump()
            print(f"Episode {episode+1} | Total Reward: {total_reward:.2f} | Steps: {steps}")
            # Save the model every 50 episodes
            if (episode + 1) % 50 == 0:
                agent.save(f"dqn_agent_episode_{episode+1}.pth")
                agent.save("dqn_agent_final.pth")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving final model...")
    finally:
        agent.save("dqn_agent_final.pth")
        pygame.quit()

if __name__ == "__main__":
    main()
