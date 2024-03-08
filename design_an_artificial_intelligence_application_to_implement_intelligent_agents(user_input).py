import random

class Agent:
    def __init__(self, name, position):
        self.name = name
        self.position = position

    def move(self, direction):
        x, y = self.position
        if direction == 'up':
            self.position = (x, y + 1)
        elif direction == 'down':
            self.position = (x, y - 1)
        elif direction == 'left':
            self.position = (x - 1, y)
        elif direction == 'right':
            self.position = (x + 1, y)

class Environment:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.agents = []

    def add_agent(self, agent):
        self.agents.append(agent)

    def display(self):
        matrix = [['.' for _ in range(self.width)] for _ in range(self.height)]
        for agent in self.agents:
            x, y = agent.position
            matrix[y][x] = 'A'  # A for Agent

        for row in matrix:
            print(' '.join(row))
        print()

def main():
    width = int(input("Enter the width of the environment: "))
    height = int(input("Enter the height of the environment: "))
    env = Environment(width=width, height=height)

    num_agents = int(input("Enter the number of agents: "))
    for i in range(1, num_agents + 1):
        agent_name = f"Agent{i}"
        initial_x = int(input(f"Enter the initial x-coordinate for {agent_name}: "))
        initial_y = int(input(f"Enter the initial y-coordinate for {agent_name}: "))
        agent = Agent(name=agent_name, position=(initial_x, initial_y))
        env.add_agent(agent)

    num_iterations = int(input("Enter the number of iterations: "))
    for _ in range(num_iterations):
        env.display()
        print("Agents' positions:")
        for agent in env.agents:
            print(f"{agent.name}: {agent.position}")
        print()

        for agent in env.agents:
            direction = random.choice(['up', 'down', 'left', 'right'])
            agent.move(direction)

if __name__ == "__main__":
    main()

# Enter the width of the environment: 5
# Enter the height of the environment: 5
# Enter the number of agents: 2
# Enter the initial x-coordinate for Agent1: 2
# Enter the initial y-coordinate for Agent1: 3
# Enter the initial x-coordinate for Agent2: 4
# Enter the initial y-coordinate for Agent2: 1
# Enter the number of iterations: 3
# . . . . .
# . . . . .
# . . A . .
# . . . . .
# . . . A .

# Agents' positions:
# Agent1: (2, 3)
# Agent2: (4, 1)

# . . . . .
# . . . . .
# . . . A .
# . . . . .
# . . A . .

# Agents' positions:
# Agent1: (2, 4)
# Agent2: (4, 0)

# . . . . .
# . . . . .
# . . . A .
# . . A . .
# . . . . .

# Agents' positions:
# Agent1: (1, 4)
# Agent2: (4, 1)


# 1. **Agent Class**:
#     - `__init__`: Initializes an agent with a name and a position `(x, y)`.
#     - `move`: Moves the agent in a given direction ('up', 'down', 'left', 'right') by updating its position accordingly.

# 2. **Environment Class**:
#     - `__init__`: Initializes the environment with a given `width` and `height`. It also initializes an empty list to store agents.
#     - `add_agent`: Adds an agent to the environment.
#     - `display`: Displays the current state of the environment by printing a matrix representation where each cell corresponds to the position of an agent.

# 3. **Main Function**:
#     - Prompts the user to input the width and height of the environment.
#     - Creates an instance of the `Environment` class.
#     - Prompts the user to input the number of agents and their initial positions.
#     - Adds the agents to the environment.
#     - Prompts the user to input the number of iterations (time steps) for the simulation.
#     - In each iteration:
#         - Displays the current state of the environment.
#         - Prints the positions of all agents.
#         - Randomly selects a direction ('up', 'down', 'left', 'right') for each agent and moves them accordingly using the `move` method.

# 4. **Execution**:
#     - The `main` function is called when the script is executed.
#     - It initializes the environment, adds agents, and runs the simulation for the specified number of iterations.

# 5. **Random Movement**:
#     - The direction of movement for each agent is randomly chosen from the list ['up', 'down', 'left', 'right'] using `random.choice`.
