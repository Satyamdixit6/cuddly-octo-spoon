import os
import time
import random
import msvcrt  # For Windows keyboard input; comment out for macOS/Linux
# For macOS/Linux, uncomment the following line and comment out msvcrt import
# import sys, tty, termios

# Clear terminal screen
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

# Get single keypress (Windows)
def get_key():
    if os.name == 'nt':
        if msvcrt.kbhit():
            return msvcrt.getch().decode('utf-8').lower()
        return None
    else:
        # For macOS/Linux, uncomment this block and comment msvcrt import
        """
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            if sys.stdin in select.select([sys.stdin], [], [], 0.1)[0]:
                return sys.stdin.read(1).lower()
            return None
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        """
        pass

# Game settings
width = 15  # Base road width
height = 20  # Display height
car = '🚗'  # Player car
opponent_car = '🚙'  # Opponent car
obstacles = ['🚧', '🛢️']  # Obstacles
power_ups = ['⚡', '🛡️', '⭐']  # Speed boost, shield, score multiplier
fuel = '⛽'  # Fuel can
road = ['-' * width for _ in range(height)]  # Initialize road
car_pos = width // 2  # Player car position
opponents = []  # List of opponent cars [(row, col), ...]
score = 0
laps = 0
lap_length = 50  # Frames per lap
frame_count = 0
speed = 0.15  # Initial game speed
power_up = None  # Current power-up
shield_active = False  # Shield status
score_multiplier = 1  # Score multiplier
fuel_level = 100  # Fuel percentage
rain = False  # Weather effect
rain_duration = 0
high_score = 0
current_width = width  # Current road width

# Main game loop
def game_loop():
    global car_pos, score, laps, frame_count, speed, power_up, shield_active, score_multiplier, fuel_level, rain, rain_duration, high_score, current_width, opponents
    while True:
        clear_screen()
        # Adjust road width randomly
        if random.random() < 0.05:  # 5% chance to change width
            current_width = random.randint(10, 18)
            road[-1] = '-' * current_width + '█' * (width - current_width)

        # Generate new road line with obstacles, power-ups, fuel, or opponents
        new_line = '-' * current_width + '█' * (width - current_width)
        if random.random() < 0.15:  # 15% chance of obstacle
            obs_pos = random.randint(0, current_width - 1)
            new_line = new_line[:obs_pos] + random.choice(obstacles) + new_line[obs_pos + 1:]
        elif random.random() < 0.05:  # 5% chance of power-up
            pu_pos = random.randint(0, current_width - 1)
            new_line = new_line[:pu_pos] + random.choice(power_ups) + new_line[pu_pos + 1:]
        elif random.random() < 0.05:  # 5% chance of fuel
            fuel_pos = random.randint(0, current_width - 1)
            new_line = new_line[:fuel_pos] + fuel + new_line[fuel_pos + 1:]
        elif random.random() < 0.03 and len(opponents) < 3:  # 3% chance of new opponent
            opp_pos = random.randint(0, current_width - 1)
            opponents.append((0, opp_pos))  # Add opponent at top

        road.pop(0)  # Remove top line
        road.append(new_line)  # Add new line at bottom

        # Move opponents
        new_opponents = []
        for row, col in opponents:
            if row + 1 < height - 1:  # Move down, stop before player row
                new_opponents.append((row + 1, col))
        opponents = new_opponents

        # Check for collision or collection
        bottom_item = road[-1][car_pos]
        if bottom_item in obstacles and not shield_active:
            print("CRASH! Game Over! Score:", score, "Laps:", laps)
            print("High Score:", high_score)
            print("Press Q to quit.")
            while True:
                key = get_key()
                if key == 'q':
                    return
                time.sleep(0.1)
        elif bottom_item in power_ups:
            power_up = bottom_item
            score += 50 * score_multiplier
            print("Power-up collected!", power_up)
        elif bottom_item == fuel:
            fuel_level = min(100, fuel_level + 20)
            score += 20 * score_multiplier
            print("Fuel collected! Fuel:", fuel_level)

        # Check for opponent collision
        for row, col in opponents:
            if row == height - 1 and col == car_pos and not shield_active:
                print("COLLISION with opponent! Game Over! Score:", score, "Laps:", laps)
                print("High Score:", high_score)
                print("Press Q to quit.")
                while True:
                    key = get_key()
                    if key == 'q':
                        return
                    time.sleep(0.1)

        # Handle input
        key = get_key()
        if key == 'q':
            print("Thanks for playing! Final Score:", score, "Laps:", laps)
            print("High Score:", high_score)
            return
        elif key == 'a' and car_pos > 0:
            car_pos -= 1
            print("SCREECH!")
        elif key == 'd' and car_pos < current_width - 1:
            car_pos += 1
            print("SCREECH!")
        elif key == ' ' and power_up:  # Use power-up
            if power_up == '⚡':
                speed = max(0.05, speed - 0.05)
                print("VROOM! Speed boost activated!")
            elif power_up == '🛡️':
                shield_active = True
                print("SHIELD ON!")
            elif power_up == '⭐':
                score_multiplier = 2
                print("Score multiplier x2 activated!")
            power_up = None
            time.sleep(0.5)

        # Update game state
        frame_count += 1
        score += score_multiplier
        fuel_level -= 0.5  # Fuel decreases over time
        if fuel_level <= 0:
            print("OUT OF FUEL! Game Over! Score:", score, "Laps:", laps)
            print("High Score:", high_score)
            print("Press Q to quit.")
            while True:
                key = get_key()
                if key == 'q':
                    return
                time.sleep(0.1)

        if frame_count >= lap_length:
            laps += 1
            frame_count = 0
            speed = max(0.05, speed - 0.02)
            print("Lap", laps, "completed! Speed increased!")
            time.sleep(1)

        if shield_active and random.random() < 0.1:
            shield_active = False
            print("Shield deactivated!")
        if score_multiplier > 1 and random.random() < 0.05:
            score_multiplier = 1
            print("Score multiplier reset!")

        # Weather effects
        if random.random() < 0.02 and not rain:
            rain = True
            rain_duration = 20
            speed += 0.05  # Slow down in rain
            print("RAIN! Drive carefully!")
        if rain:
            rain_duration -= 1
            if rain_duration <= 0:
                rain = False
                speed = max(0.05, speed - 0.05)
                print("Rain cleared!")

        high_score = max(high_score, score)

        # Display game
        print("Terminal Grand Prix: Turbo Edition! Score:", score, "Laps:", laps, "High Score:", high_score)
        print("Controls: A (left), D (right), Space (use power-up), Q (quit)")
        print(f"Power-up: {power_up if power_up else 'None'} | Shield: {'ON' if shield_active else 'OFF'} | Fuel: {fuel_level:.1f}% | Weather: {'Rain' if rain else 'Clear'} | Multiplier: x{score_multiplier}")
        print()
        for i, line in enumerate(road[:-1]):
            display_line = list(line)
            for row, col in opponents:
                if row == i and col < len(display_line):
                    display_line[col] = opponent_car
            print('|' + ''.join(display_line) + '|')
        bottom_line = '-' * current_width + '█' * (width - current_width)
        bottom_line = bottom_line[:car_pos] + car + bottom_line[car_pos + 1:]
        print('|' + bottom_line + '|')
        
        time.sleep(speed)

# Start the game
if __name__ == "__main__":
    print("Starting Terminal Grand Prix: Turbo Edition...")
    print("Race against opponents, collect fuel and power-ups, and survive the weather!")
    time.sleep(2)
    game_loop()