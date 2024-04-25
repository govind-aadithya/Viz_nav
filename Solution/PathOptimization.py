import networkx as nx
import matplotlib.pyplot as plt
import heapq
import numpy as np

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def get_neighbors(point, maze_points):
    neighbors = []
    
    # Include only horizontal and vertical neighbors
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        neighbor = (point[0] + dx, point[1] + dy)
        if neighbor in maze_points:
            neighbors.append(neighbor)
    
    return neighbors

def astar(maze_points, start, end):
    open_set = []
    closed_set = set()
    heapq.heappush(open_set, (0, start))
    came_from = {}

    g_score = {point: float('inf') for point in maze_points}
    g_score[start] = 0

    f_score = {point: float('inf') for point in maze_points}
    f_score[start] = heuristic(start, end)

    while open_set:
        current = heapq.heappop(open_set)[1]

        if current == end:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        closed_set.add(current)

        for neighbor in get_neighbors(current, maze_points):
            if neighbor in closed_set:
                continue

            tentative_g_score = g_score[current] + 1

            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, end)
                if neighbor not in open_set:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    print("No path found between start and end points.")
    return None

def calculate_path(maze_points = [(250, 250), (251, 250), (252, 250), (253, 250), (254, 250), (255, 250), (256, 250), (257, 250), (258, 250), (259, 250), (260, 250), (261, 250), (262, 250), (263, 250), (264, 250), (265, 250), (266, 250), (267, 250), (268, 250), (269, 250), (270, 250), (271, 250), (272, 250), (273, 250), (274, 250), (275, 250), (276, 250), (277, 250), (278, 250), (279, 250), (280, 250), (281, 250), (282, 250), (283, 250), (284, 250), (285, 250), (286, 250), (287, 250), (288, 250), (289, 250), (290, 250), (291, 250), (292, 250), (293, 250), (294, 250), (295, 250), (296, 250), (297, 250), (298, 250), (299, 250), (300, 250), (301, 250), (302, 250), (303, 250), (303, 251), (303, 252), (303, 253), (303, 254), (303, 255), (303, 256), (303, 257), (303, 258), (303, 259), (303, 260), (303, 261), (303, 262), (304, 262), (305, 262), (306, 262), (307, 262), (308, 262), (309, 262), (310, 262), (311, 262), (312, 262), (313, 262), (314, 262), (315, 262), (316, 262), (317, 262), (318, 262), (318, 263), (318, 264), (318, 265), (318, 266), (318, 267), (318, 268), (318, 269), (318, 270), (318, 271), (318, 272), (318, 273), (318, 274), (318, 275), (301, 276), (302, 276), (303, 276), (304, 276), (305, 276), (306, 276), (307, 276), (308, 276), (309, 276), (310, 276), (311, 276), (312, 276), (313, 276), (314, 276), (315, 276), (316, 276), (317, 276), (318, 276), (301, 277), (301, 278), (301, 279), (301, 280), (301, 281), (301, 282), (301, 283), (301, 284), (301, 285), (301, 286), (301, 287), (301, 288), (301, 289), (301, 290), (301, 291), (301, 292), (301, 293), (301, 294), (301, 295), (301, 296), (301, 297), (301, 298), (301, 299), (301, 300), (301, 301), (301, 302), (301, 303), (286, 304), (287, 304), (288, 304), (289, 304), (290, 304), (291, 304), (292, 304), (293, 304), (294, 304), (295, 304), (296, 304), (297, 304), (298, 304), (299, 304), (300, 304), (301, 304)], end_point = (301, 304)):
    # Given maze intermediate points
    #print(maze_points)
    # Start and end points
    start_point = maze_points[0]

    #print("start = ", start_point)
    #print("Target Frame Location = ", end_point)
    # Find the shortest path
    shortest_path = astar(maze_points, start_point, end_point)
    shortest_path_new = [(shortest_path[0][0],shortest_path[0][1],0)]
    for i in range(len(shortest_path)-1):
        slope = np.degrees(np.arctan2((shortest_path[i+1][1] - shortest_path[i][1]), (shortest_path[i+1][0] - shortest_path[i][0])))
        if slope == -90:
            slope = 270
        shortest_path_new.append((shortest_path[i+1][0],shortest_path[i+1][1],slope))    

    '''    # Print the result
    if shortest_path_new:
        print(f"Shortest Path: {shortest_path_new}")'''

    maze_points = np.array(maze_points)
    shortest_path = np.array(shortest_path)

    '''
    plt.scatter(maze_points[:,1],maze_points[:,0])

    # Plot shortest path with increased thickness (linewidth=2 in this case)
    plt.plot( shortest_path[:,1], shortest_path[:,0], "r", linewidth=2)

    # Show the plot
    plt.show()    
    '''
    
    return shortest_path_new
