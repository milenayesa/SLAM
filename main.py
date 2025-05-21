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



def generate_room(size, num_obstacles=30, num_corridors=4):
    width, height = size
    room = np.zeros((width, height), dtype=int)

    for _ in range(num_obstacles):
        x = random.randint(1, width - 2)
        y = random.randint(1, height - 2)
        room[x, y] = 1

    for _ in range(3):
        x = random.randint(1, width - 5)   
        y = random.randint(1, height - 4)  

        room[x:x+4, y] = 1
        room[x:x+4, y+2] = 1

    return room



class Drone:
    def __init__(self, room, start):
        self.room = room
        self.x, self.y = start
        self.path = [start]
        self.visited = np.zeros(room.shape, dtype=bool)
        self.visited[start] = True
        self.planned_path = []
        self.built_map = np.full(room.shape, -1, dtype=float)
        self.checked_map = np.zeros(room.shape, dtype=bool)
        self.built_map[start] = 0
        self.checked_map[start] = True
        self.lidar_range = 5
        self.pose_graph = nx.Graph()
        self.pose_counter = 0
        self.pose_graph.add_node(self.pose_counter, pos=start)
        self.lidar_points = []

    def move_to(self, next_pos):
        self.x, self.y = next_pos
        self.path.append(next_pos)
        self.visited[next_pos] = True
        self.pose_counter += 1
        self.pose_graph.add_node(self.pose_counter, pos=next_pos)
        self.pose_graph.add_edge(self.pose_counter - 1, self.pose_counter)

    def plan_next_goal(self):
        width, height = self.built_map.shape
        potential_goals = []
        for x in range(width):
            for y in range(height):
                if self.built_map[x, y] == 0:  
                    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < width and 0 <= ny < height:
                            if self.built_map[nx, ny] == -1:  
                                potential_goals.append((x, y))
                                break

        potential_goals.sort(key=lambda c: abs(c[0] - self.x) + abs(c[1] - self.y))
        for goal in potential_goals:
            path = astar(self.room, (self.x, self.y), goal)
            if path:
                return goal
        return None


    def plan_path_to_goal(self, goal):
        return astar(self.room, (self.x, self.y), goal)

    def follow_planned_path(self):
        if self.planned_path:
            self.move_to(self.planned_path.pop(0))

    def scan_environment(self):
        self.lidar_points = []
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1),  
                      (1, 1), (-1, -1), (1, -1), (-1, 1)]  

        for dx, dy in directions:
            noisy_range = max(1, self.lidar_range + np.random.normal(0, 0.4))  

            for r in range(1, int(noisy_range) + 1):  
                lx = self.x + dx * r
                ly = self.y + dy * r

                if 0 <= lx < self.room.shape[0] and 0 <= ly < self.room.shape[1]:
                    if self.room[lx, ly] == 1:  
                        self.built_map[lx, ly] = 1
                        self.checked_map[lx, ly] = True
                        break

                    self.lidar_points.append((lx, ly)) 

                    if self.room[lx, ly] == 0 and not self.checked_map[lx, ly]:
                        self.built_map[lx, ly] = 0
                        self.checked_map[lx, ly] = True


def visualize_step(room, drone, axs):
    axs[0].clear()
    axs[1].clear()
    axs[0].imshow(room.T, cmap="gray_r", origin='lower')
    rp = np.array(drone.path)
    axs[0].plot(rp[:, 0] + 0.5, rp[:, 1] + 0.5, 'b.-')
    axs[0].scatter(drone.x + 0.5, drone.y + 0.5, c="red", s=80)
    if drone.planned_path:
        pp = np.array(drone.planned_path)
        axs[0].plot(pp[:, 0] + 0.5, pp[:, 1] + 0.5, 'k--')
    if drone.lidar_points:
        lp = np.array(drone.lidar_points)
        axs[0].scatter(lp[:, 0] + 0.5, lp[:, 1] + 0.5, c='yellow', s=10)

    cmap = ListedColormap(['white', 'gray', 'black'])
    bounds = [-1.5, -0.1, 0.1, 1.5]
    norm = BoundaryNorm(bounds, cmap.N)
    axs[1].imshow(drone.built_map.T, cmap=cmap, norm=norm, origin='lower')

    axs[0].set_title("Room & Path")
    axs[1].set_title("Built Map")
    plt.pause(0.1)

room = generate_room((20, 20))
start = (10, 10) if room[10, 10] == 0 else tuple(np.argwhere(room == 0)[0])
drone = Drone(room, start)

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
max_iter = 5000
iteration = 0

while iteration < max_iter:
    drone.scan_environment()
    if not drone.planned_path:
        goal = drone.plan_next_goal()
        if goal:
            path = drone.plan_path_to_goal(goal)
            if path is not None:
                drone.planned_path = path
                drone.planned_path.pop(0) 
            else:
                drone.planned_path = []
    drone.follow_planned_path()
    if iteration % 5 == 0:
        visualize_step(room, drone, axs)
    iteration += 1


print("Exploration complete.")
plt.show()
