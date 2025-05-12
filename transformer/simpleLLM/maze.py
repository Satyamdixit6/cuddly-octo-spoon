import os
import time

# Clear terminal screen
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

# Maze layout (2D grid, # = wall, . = path, $ = treasure)
maze = [
    ['#', '#', '#', '#', '#', '#'],
    ['#', '.', '.', '.', '#', '#'],
    ['#', '.', '#', '.', '#', '#'],
    ['#', '.', '#', '.', '.', '$'],
    ['#', '#', '#', '#', '#', '#']
]

# Player position and direction
player_x, player_y = 1, 1  # Starting position
player_dir = 'N'  # Facing North (N, E, S, W)

# Directions for movement (dx, dy for North, East, South, West)
directions = {'N': (-1, 0), 'E': (0, 1), 'S': (1, 0), 'W': (0, -1)}
dir_order = ['N', 'E', 'S', 'W']

# Render a simple 3D view based on player direction
def render_view(x, y, direction):
    view = ""
    if direction == 'N':
        steps = [(x-1, y), (x-2, y), (x-3, y)]
    elif direction == 'E':
        steps = [(x, y+1), (x, y+2), (x, y+3)]
    elif direction == 'S':
        steps = [(x+1, y), (x+2, y), (x+3, y)]
    else:  # West
        steps = [(x, y-1), (x, y-2), (x, y-3)]

    for i, (nx, ny) in enumerate(steps):
        if 0 <= nx < len(maze) and 0 <= ny < len(maze[0]):
            if maze[nx][ny] == '#':
                view += "███" if i == 0 else "██" if i == 1 else "█"
            elif maze[nx][ny] == '$':
                view += "$$$" if i == 0 else "$$" if i == 1 else "$"
            else:
                view += "..." if i == 0 else ".." if i == 1 else "."
        else:
            view += "███" if i == 0 else "██" if i == 1 else "█"
        view += "\n"
    return view

# Print the maze with player position
def print_maze(x, y):
    for i in range(len(maze)):
        for j in range(len(maze[0])):
            if i == x and j == y:
                print('P', end=' ')
            else:
                print(maze[i][j], end=' ')
        print()
    print(f"Facing: {player_dir}")

# Main game loop
def game_loop():
    global player_x, player_y, player_dir
    while True:
        clear_screen()
        print("3D Maze Game! Find the treasure ($)!")
        print("Controls: W (forward), A (turn left), S (backward), D (turn right), Q (quit)")
        print("\n3D View:")
        print(render_view(player_x, player_y, player_dir))
        print("\nMap:")
        print_maze(player_x, player_y)

        move = input("Your move: ").lower()
        new_x, new_y = player_x, player_y
        if move == 'q':
            print("Thanks for playing!")
            break
        elif move == 'a':  # Turn left
            player_dir = dir_order[(dir_order.index(player_dir) - 1) % 4]
        elif move == 'd':  # Turn right
            player_dir = dir_order[(dir_order.index(player_dir) + 1) % 4]
        elif move == 'w':  # Move forward
            dx, dy = directions[player_dir]
            new_x, new_y = player_x + dx, player_y + dy
        elif move == 's':  # Move backward
            dx, dy = directions[player_dir]
            new_x, new_y = player_x - dx, player_y - dy

        # Check if move is valid
        if 0 <= new_x < len(maze) and 0 <= new_y < len(maze[0]) and maze[new_x][new_y] != '#':
            player_x, player_y = new_x, new_y
            if maze[player_x][player_y] == '$':
                clear_screen()
                print("Congratulations! You found the treasure!")
                break
        else:
            print("Can't move there! Try again.")
            time.sleep(1)

# Start the game
if __name__ == "__main__":
    game_loop()