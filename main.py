import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
import heapq
from matplotlib.colors import ListedColormap, BoundaryNorm


def astar(room, start, goal):

    width, height = room.shape

    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), 0, start, [start]))
    visited = set()

    while open_set:
        est, cost, current, path = heapq.heappop(open_set)
        if current == goal:
            return path
        if current in visited:
            continue
        visited.add(current)
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < width and 0 <= neighbor[1] < height:
                if room[neighbor] == 0 and neighbor not in visited:
                    new_cost = cost + 1
                    est_new = new_cost + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (est_new, new_cost, neighbor, path + [neighbor]))
    return None


def generate_room(size, num_obstacles=30):
    
    width, height = size
    room = np.zeros((width, height), dtype=int)

    for _ in range(num_obstacles):
        x = random.randint(1, width - 2)
        y = random.randint(1, height - 2)
        room[x, y] = 1

    num_closed_objects = 2  
    
    for _ in range(num_closed_objects):
        obj_width = random.randint(3, 6)
        obj_height = random.randint(3, 6)

        max_x = width - obj_height - 1
        max_y = height - obj_width - 1
        if max_x < 1 or max_y < 1:
            continue  
        top_left_x = random.randint(1, max_x)
        top_left_y = random.randint(1, max_y)

        room[top_left_x, top_left_y:top_left_y + obj_width] = 1  # Top border.
        room[top_left_x + obj_height - 1, top_left_y:top_left_y + obj_width] = 1  # Bottom border.
        room[top_left_x:top_left_x + obj_height, top_left_y] = 1  # Left border.
        room[top_left_x:top_left_x + obj_height, top_left_y + obj_width - 1] = 1  # Right border.

        
        room[top_left_x + 1:top_left_x + obj_height - 1, top_left_y + 1:top_left_y + obj_width - 1] = 2

    return room


class Drone:

    def __init__(self, room, start):
        self.room = room
        self.x, self.y = start  # Current cell
        self.path = [start]  # Actual path.
        self.visited = np.zeros(room.shape, dtype=bool)
        self.visited[start] = True
        self.planned_path = []  # Current planned route (list of cells).
        self.built_map = np.full(room.shape, -1, dtype=int) # Build map from LiDAR scans.
        self.lidar_range = 5  # Maximum LiDAR range.

        # Pose graph (for illustration).
        self.pose_graph = nx.Graph()
        self.pose_counter = 0
        self.pose_graph.add_node(self.pose_counter, pos=start)

    def get_free_neighbors(self):

        neighbors = []
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx_cell = self.x + dx
            ny_cell = self.y + dy
            if (0 <= nx_cell < self.room.shape[0]
                    and 0 <= ny_cell < self.room.shape[1]
                    and self.room[nx_cell, ny_cell] == 0):
                neighbors.append((nx_cell, ny_cell))
        return neighbors

    def move_to(self, next_pos):

        self.x, self.y = next_pos
        self.path.append(next_pos)
        self.visited[next_pos] = True
        self.pose_counter += 1
        self.pose_graph.add_node(self.pose_counter, pos=next_pos)
        self.pose_graph.add_edge(self.pose_counter - 1, self.pose_counter, weight=1)

    def plan_next_goal(self):

        width, height = self.room.shape
        candidates = [(i, j) for i in range(width) for j in range(height)
            if self.room[i, j] == 0 and not self.visited[i, j]]
        if candidates:
            curr = (self.x, self.y)
            candidates.sort(key=lambda cell: abs(cell[0] - curr[0]) + abs(cell[
                1] - curr[1]))
            return candidates[0]
        free_cells = [(i, j) for i in range(width) for j in range(height)
            if self.room[i, j] == 0 and (i, j) != (self.x, self.y)]
        if free_cells:
            return random.choice(free_cells)
        return None

    def plan_path_to_goal(self, goal):
        start = (self.x, self.y)
        return astar(self.room, start, goal)

    def follow_planned_path(self):
        
        if self.planned_path and len(self.planned_path) > 1:
            self.planned_path.pop(0)
            next_pos = self.planned_path[0]
            self.move_to(next_pos)
            return

        goal = self.plan_next_goal()
        if goal is not None:
            new_path = self.plan_path_to_goal(goal)
            if new_path is not None and len(new_path) > 1:
                self.planned_path = new_path
                self.planned_path.pop(0)
                next_pos = self.planned_path[0]
                self.move_to(next_pos)
                return

        # Fall back: choose a random free neighbor.
        neighbors = self.get_free_neighbors()
        if neighbors:
            next_pos = random.choice(neighbors)
            self.planned_path = [(self.x, self.y), next_pos]
            self.move_to(next_pos)

    def scan_environment(self):
        
        measurements = []
        angles = [0, 90, 180, 270]
        for angle in angles:
            hit_obstacle = False
            for r in range(1, self.lidar_range + 1):
                lx = self.x + int(round(np.cos(np.radians(angle)) * r))
                ly = self.y + int(round(np.sin(np.radians(angle)) * r))
                if 0 <= lx < self.room.shape[0] and 0 <= ly < self.room.shape[1]:
                    if self.room[lx, ly] != 0:
                        self.built_map[lx, ly] = 1
                        measurements.append((lx, ly))
                        hit_obstacle = True
                        break
                    else:
                        self.built_map[lx, ly] = 0
                else:
                    break
            if not hit_obstacle:
                for r in range(1, self.lidar_range + 1):
                    lx = self.x + int(round(np.cos(np.radians(angle)) * r))
                    ly = self.y + int(round(np.sin(np.radians(angle)) * r))
                    if 0 <= lx < self.room.shape[0] and 0 <= ly < self.room.shape[1]:
                        self.built_map[lx, ly] = 0
        return measurements



def visualize(room, drone):
    
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Display room layout and drone path.
    axs[0].imshow(room.T,
                  cmap="gray_r",
                  origin='lower',
                  extent=[0, room.shape[0], 0, room.shape[1]])
    rp = np.array(drone.path)
    if rp.shape[0] > 0:
        axs[0].plot(rp[:, 0] + 0.5,
                    rp[:, 1] + 0.5,
                    'b.-',
                    label="Drone Path",
                    linewidth=2)
    axs[0].scatter(drone.x + 0.5, drone.y + 0.5, c="red", s=80, label="Drone")
    if drone.planned_path and len(drone.planned_path) > 1:
        pp = np.array(drone.planned_path)
        axs[0].plot(pp[:, 0] + 0.5,
                    pp[:, 1] + 0.5,
                    'k--',
                    label="Planned Route",
                    linewidth=2)
    axs[0].set_xlim(0, room.shape[0])
    axs[0].set_ylim(0, room.shape[1])
    axs[0].set_title("Room Map & Drone Trajectory")
    axs[0].legend(loc="upper right")

    # Right: Display the built occupancy grid.
    cmap_built = ListedColormap(['lightblue', 'white', 'black'])
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = BoundaryNorm(bounds, cmap_built.N)
    axs[1].imshow(drone.built_map.T,
                  cmap=cmap_built,
                  norm=norm,
                  origin='lower',
                  extent=[0, room.shape[0], 0, room.shape[1]])
    axs[1].set_title("Built Map via LiDAR Scans")

    # Mark the start of the drone's path with a distinct color (green for example).
    start_pos = drone.path[0] if drone.path else (drone.x, drone.y)  # Get the start position
    axs[0].scatter(start_pos[0] + 0.5,
                   start_pos[1] + 0.5,
                   c="green",
                   s=100,
                   label="Start",
                   edgecolors="black",
                   marker="o")

    plt.tight_layout()
    plt.show()


#main

room_size = (20, 20)
room = generate_room(room_size, num_obstacles=30)

# Select a valid starting position (must be a free cell)
default_start = (room_size[0] // 2, room_size[1] // 2)
if room[default_start] != 0:
    free_cells = [(i, j) for i in range(room_size[0])
                  for j in range(room_size[1]) if room[i, j] == 0]
    start = random.choice(free_cells) if free_cells else default_start
else:
    start = default_start

print(f"Drone starting at {start}")
drone = Drone(room, start)

# Count the number of free (navigable) cells
total_free = np.sum(room == 0)

# Begin exploration and mapping
iteration = 0
max_iterations = 10000  # Safety limit
while np.sum(np.logical_and(drone.visited, room == 0)) < total_free and iteration < max_iterations:
    drone.follow_planned_path()
    drone.scan_environment()
    iteration += 1

    if iteration % 100 == 0:
        visited = np.sum(np.logical_and(drone.visited, room == 0))
        print(f"Iteration {iteration}: {visited}/{total_free} free cells visited.")

print(f"Mapping complete after {iteration} iterations.")
visualize(room, drone)
