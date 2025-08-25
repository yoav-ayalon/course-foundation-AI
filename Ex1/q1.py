import csv
import os
import random
from collections import deque
import math
import numpy as np
from scipy.optimize import linear_sum_assignment

neighbors = {
    'AK': [],
    'AL': ['GA', 'TN', 'MS', 'FL'],
    'AR': ['MO', 'TN', 'MS', 'LA', 'TX', 'OK'],
    'AS': [],
    'AZ': ['CA', 'NV', 'UT', 'CO', 'NM'],
    'CA': ['OR', 'NV', 'AZ'],
    'CO': ['WY', 'NE', 'KS', 'OK', 'NM', 'UT'],
    'CT': ['MA', 'NY', 'RI'],
    'DC': [],
    'DE': ['MD', 'PA', 'NJ'],
    'FL': ['AL', 'GA'],
    'GA': ['TN', 'NC', 'SC', 'FL', 'AL'],
    'GU': [],
    'HI': [],
    'IA': ['WI', 'IL', 'MO', 'NE', 'SD', 'MN'],
    'ID': ['MT', 'WY', 'UT', 'NV', 'OR', 'WA'],
    'IL': ['WI', 'IA', 'MO', 'KY', 'IN', 'MI'],
    'IN': ['MI', 'OH', 'KY', 'IL'],
    'KS': ['NE', 'MO', 'OK', 'CO'],
    'KY': ['IL', 'IN', 'OH', 'WV', 'VA', 'TN', 'MO'],
    'LA': ['AR', 'MS', 'TX'],
    'MA': ['NH', 'VT', 'NY', 'CT', 'RI'],
    'MD': ['PA', 'DE', 'VA', 'WV'],
    'ME': ['NH'],
    'MI': ['WI', 'IL', 'IN', 'OH'],
    'MN': ['WI', 'IA', 'SD', 'ND'],
    'MO': ['IA', 'IL', 'KY', 'TN', 'AR', 'OK', 'KS', 'NE'],
    'MP': [],
    'MS': ['TN', 'AL', 'LA'],
    'MT': ['ID', 'WY', 'SD', 'ND'],
    'NC': ['VA', 'TN', 'GA', 'SC'],
    'ND': ['MN', 'SD', 'MT'],
    'NE': ['SD', 'IA', 'MO', 'KS', 'CO', 'WY'],
    'NH': ['VT', 'ME', 'MA'],
    'NJ': ['NY', 'PA', 'DE'],
    'NM': ['AZ', 'CO', 'OK', 'TX'],
    'NV': ['OR', 'ID', 'UT', 'AZ', 'CA'],
    'NY': ['VT', 'MA', 'CT', 'PA', 'NJ'],
    'OH': ['MI', 'IN', 'KY', 'WV', 'PA'],
    'OK': ['KS', 'MO', 'AR', 'TX', 'NM', 'CO'],
    'OR': ['WA', 'ID', 'NV', 'CA'],
    'PA': ['NY', 'OH', 'WV', 'MD', 'DE', 'NJ'],
    'PR': [],
    'RI': ['MA', 'CT'],
    'SC': ['NC', 'GA'],
    'SD': ['ND', 'MN', 'IA', 'NE', 'MT', 'WY'],
    'TN': ['KY', 'VA', 'NC', 'GA', 'AL', 'MO', 'AR', 'MS'],
    'TX': ['OK', 'AR', 'LA', 'NM'],
    'UT': ['ID', 'WY', 'CO', 'NM', 'AZ', 'NV'],
    'VA': ['WV', 'MD', 'NC', 'TN', 'KY'],
    'VI': [],
    'VT': ['NY', 'MA', 'NH'],
    'WA': ['ID', 'OR'],
    'WI': ['MI', 'IL', 'IA', 'MN'],
    'WV': ['OH', 'PA', 'MD', 'VA', 'KY'],
    'WY': ['MT', 'SD', 'NE', 'CO', 'UT', 'ID']
}


class Location:
    def __init__(self, name, state):
        self.name = name
        self.state = state
        self.neighbors = []
        self.visited = False
        self.parent = None
        self.starting = False
        self.goal = False
        self.color = None
        self.g_score = float('inf')
        self.h_score = float('inf')
        self.f_score = float('inf')

    def add_neighbor(self, neighbor):
        if neighbor not in self.neighbors and (neighbor.name, neighbor.state) != (self.name, self.state):
            self.neighbors.append(neighbor)

    def calculate_f_score(self):
        self.f_score = self.g_score + self.h_score

    def __repr__(self):
        return (
            f"Location(name={self.name}, state={self.state}, color={self.color}, starting={self.starting}, goal={self.goal}, "
            f"visited={self.visited}, g_score={self.g_score}, h_score={self.h_score}, f_score={self.f_score}, neighbors={[n.name for n in self.neighbors]})")

    def print_neighbors(self):
        print(f"Neighbors of {self.name}, {self.state}:")
        for neighbor in self.neighbors:
            print(f"{neighbor.name}, {neighbor.state}")

    @classmethod
    def get_location(cls, locations, name, state):
        key = f"{name}, {state}"
        return locations.get(key)


# read the csv file and make object from the locations
def read_from_csv(file_path):
    locations = {}
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)

        for row in csvreader:
            loc1, loc2 = row
            name1, state1 = loc1.split(", ")
            name2, state2 = loc2.split(", ")

            if loc1 not in locations:
                locations[loc1] = Location(name1, state1)
            if loc2 not in locations:
                locations[loc2] = Location(name2, state2)

            locations[loc1].add_neighbor(locations[loc2])
            locations[loc2].add_neighbor(locations[loc1])

    return locations  # Return as dictionary for easy lookup


# validation of the input and reset start and goal objects
def valid_location_list(graph, location_list, str):
    valid_colors = {"Blue", "Red"}
    string_data = location_list[0]
    locations = [location.strip() for location in string_data.split(';')]

    for location in locations:
        parts = location.split(", ")

        if len(parts) != 3:
            print(f"Invalid format: {location}")
            return False

        color, name, state_code = parts[0], parts[1], parts[2]

        if color not in valid_colors:
            print(f"Invalid color: {color}")
            return False

        node_name = f"{name}, {state_code}"

        if node_name not in graph:
            print(f"Invalid location: {node_name}")
            return False

        if str == "starting":
            graph[node_name].starting = True
            graph[node_name].visited = True
            graph[node_name].color = color
            graph[node_name].g_score = 0

        if str == "goal":
            graph[node_name].goal = True
            graph[node_name].color = color

    # print(str, "location list is valid")
    return True


# heuristic by BFS of states neighbors
def heuristic_calculation(start_state, end_state):
    visited = set()
    queue = deque([(start_state, 0)])  # Initialize the queue with the start state and distance 0

    while queue:
        current_state, distance = queue.popleft()

        if current_state == end_state:
            return round(math.exp(0.6 * distance) + 1.8 * distance)  # Return the distance when the end state is reached

        visited.add(current_state)

        for neighbor in neighbors.get(current_state, []):  # Retrieve neighbors from the global 'neighbors' dictionary
            if neighbor not in visited:
                queue.append((neighbor, distance + 1))

    return -1000  # no path exists between start and end state


# reset the location between running the A* again
def reset_locations(graph):
    for location in graph.items():
        location = location[1]
        location.g_score = float('inf')
        location.h_score = float('inf')
        location.f_score = float('inf')
        location.visited = False
        location.parent = None


# return the path in string
def return_path(start, goal):
    ans = []
    location = goal
    ans.append(f"{location.name}, {location.state} ({start.color[0]})")

    while location != start:
        ans.append(f"{location.parent.name}, {location.parent.state} ({start.color[0]})")
        location = location.parent

    ans.reverse()
    return ans


# return the un-visited and min h_score to next location in hill climbing method
def get_next_location_hill_climbing(current):
    unvisited_neighbors = [neighbor for neighbor in current.neighbors if not neighbor.visited and neighbor.h_score > 0]

    if unvisited_neighbors:
        unvisited_neighbors = sorted(unvisited_neighbors, key=lambda x: x.h_score)
        next_locations = unvisited_neighbors[0]
        if next_locations.h_score == unvisited_neighbors[-1].h_score:
            return random.choice(unvisited_neighbors)
        return next_locations

    else:
        return None


# return an un_visited random neighbor simulated annealing method
def get_next_location_simulated_annealing(current):
    unvisited_neighbors = [neighbor for neighbor in current.neighbors if not neighbor.visited]
    if unvisited_neighbors:
        return random.choice(unvisited_neighbors)
    return None


# return an un_visited top k neighbors k_beam method
def get_next_location_k_baem(list1, k):
    unvisited = [i for i in list1 if i.h_score > 0]
    unique = list(dict.fromkeys(unvisited))

    next_locations = []
    if unique:
        unique = random.sample(unique, len(unique))
        sorted_neighbors = sorted(unique, key=lambda x: x.h_score)
        next_locations = sorted_neighbors[:k]
        return next_locations

    return None


# 1
def A_star(graph, start, goal):
    frontier = []
    reset_locations(graph)

    start.visited = True
    start.g_score = 0
    start.f_score = heuristic_calculation(start.state, goal.state)
    frontier.append([start.f_score, start])

    while frontier:
        current = frontier.pop(0)[1]
        current.visited = True
        if current == goal:
            return True

        for neighbor in current.neighbors:
            if not neighbor.visited:
                tentative_gscore = current.g_score + 1

                if tentative_gscore < neighbor.g_score:
                    neighbor.g_score = tentative_gscore
                    neighbor.h_score = heuristic_calculation(neighbor.state, goal.state)
                    neighbor.calculate_f_score()
                    neighbor.parent = current
                    frontier.append([neighbor.f_score, neighbor])

        frontier = sorted(frontier, key=lambda x: x[0])

    return False


# 2
def hill_climbing(graph, start, goal):
    reset_locations(graph)
    restart = 5
    current = start
    for _ in range(restart):

        while current != goal:
            current.h_score = heuristic_calculation(current.state, goal.state)
            neighbors = current.neighbors

            for neighbor in neighbors:
                neighbor.h_score = heuristic_calculation(neighbor.state, goal.state)
                if neighbor == goal:
                    neighbor.visited = True
                    neighbor.parent = current
                    return True

            next_location = get_next_location_hill_climbing(current)

            if next_location == None or next_location.h_score > current.h_score:
                break

            else:
                next_location.visited = True
                next_location.parent = current
                current = next_location

        reset_locations(graph)
        current = random.choice(start.neighbors)
        start.visited = True
        current.parent = start

    return False


# 3
def simulated_annealing(graph, start, goal):
    reset_locations(graph)
    temp = 100
    current = start
    t = 1
    printed = None

    while temp > 0.1:
        current.visited = True
        next = get_next_location_simulated_annealing(current)

        if next == None:
            return False, None

        if next == goal:
            next.parent = current
            return True, printed

        current.h_score = heuristic_calculation(current.state, goal.state)
        next.h_score = heuristic_calculation(next.state, goal.state)
        delta = next.h_score - current.h_score

        if delta < 0 or random.random() < math.exp(-1 / temp):
            next.parent = current
            next.visited = True
            current = next

        if t == 2:
            if delta < 0:
                printed = f"{next.name}, {next.state}, 1"
            else:
                printed = f"{next.name}, {next.state}, {round(math.exp(-1 / temp),3)}"

        temp = 100 - math.log10(t + 1) * 50
        t += 1

    return False, None


# 4
def k_beam(graph, start, goal):
    reset_locations(graph)
    k = 3
    current = [start]
    min_h_score = float('inf')
    i = 0
    printed = []
    color = "(" + start.color[0] + ")"

    while True:
        i += 1
        neighborhood = []
        for location in current:
            neighbors = location.neighbors

            for neighbor in neighbors:
                if neighbor == goal:
                    neighbor.parent = location
                    return True, printed

                if not neighbor.visited:
                    neighbor.h_score = heuristic_calculation(neighbor.state, goal.state)
                    neighbor.parent = location
                    if neighbor.h_score <= min_h_score:
                        neighborhood.append(neighbor)

        # from next_locations choose the top 3
        current = get_next_location_k_baem(neighborhood, k)
        if current == None:
            break

        if i == 2:
            for p in current:
                printed.extend([p.name, p.state, color])

        for j in current:  # error - 'NoneType' object is not iterable
            j.visited = True
            if j.h_score < min_h_score:
                min_h_score = j.h_score

    return False, None


# 5
def genetic_algorithm(graph, start, goal):
    reset_locations(graph)

    def check_all_same(population):
        for item in population:
            if item != population[0]:
                return False
        return True

    def find_location(graph, neighbor):
        neighbor_parts = neighbor.split(', ')

        # Extract the county name and state abbreviation
        neighbor_name = neighbor_parts[0]
        neighbor_state = neighbor_parts[1].split()[0]

        for key, location in graph.items():
            if location.name == neighbor_name and location.state == neighbor_state:
                return location
        return None

    def mutate_kid_in_probability(kid):
        color = kid[0][-3:]
        if random.random() < 0.4:
            mutate_index = random.randint(1, len(kid) - 2)
            tries = 0

            while tries < 3:
                current_node = kid[mutate_index]
                neighbors_before = kid[mutate_index - 1]
                neighbors_after = kid[mutate_index + 1]

                before = find_location(graph, neighbors_before)
                after = find_location(graph, neighbors_after)

                before_neigbors = set(before.neighbors)
                after_neighbors = set(after.neighbors)

                common_neighbors = before_neigbors.intersection(after_neighbors)

                if common_neighbors:
                    new_county = random.choice(list(common_neighbors))
                    kid[mutate_index] = f"{new_county.name}, {new_county.state} " + color
                    break
                else:
                    mutate_index = random.randint(1, len(kid) - 2)
                    tries += 1

    # search if there is a path
    if not A_star(graph, start, goal):
        return False, None

    # initial population using k-beam
    population = []
    while len(population) < 10:
        entity, _ = k_beam(graph, start, goal)
        if entity:
            path = return_path(start, goal)
            population.append(path)

    unchanged_iterations = 0
    previous_min_length = None
    generation = 0
    printed = []

    while unchanged_iterations < 70:
        # print('gen', generation)

        # Calculate max length
        max_length = max(len(item) for item in population)

        # Calculate min length
        min_length = min(len(item) for item in population)

        # Calculate mean length
        mean_length = sum(len(item) for item in population) / len(population)

        # Print statistics
        # print(f"Max Length: {max_length}")
        # print(f"Min Length: {min_length}")
        # print(f"Mean Length: {mean_length:.2f}")

        if check_all_same(population):
            return population[0], printed

        population = sorted(population, key=len)
        len_list = list(reversed([len(entity) for entity in population]))
        total_length = sum(len_list)
        fitness_list = [len_list / total_length for len_list in len_list]  # the probibility/ grade of the path (sum=1)

        chosen_tuples = []
        for _ in range(5):
            while True:
                # Randomly choose two entities based on their probabilities
                entity1 = random.choices(population, weights=fitness_list, k=1)[0]
                entity2 = random.choices(population, weights=fitness_list, k=1)[0]

                # Ensure the two entities in a tuple are not the same
                if entity1 != entity2:
                    chosen_tuples.append((entity1, entity2))
                    break

        new_population = []
        for tuple in chosen_tuples:
            parent1, parent2 = tuple
            common_node_index1 = None
            common_node_index2 = None
            is_common_node = False

            # looking for common location in the tuple
            for i in range(1, len(parent1)):
                for j in range(len(parent2) - 2, -1, -1):
                    if parent2[j] == parent1[i]:
                        common_node_index1 = i
                        common_node_index2 = j
                        is_common_node = True
                        break

            # Create kids
            if is_common_node:
                kid1 = parent1[:common_node_index1] + parent2[common_node_index2:]
                kid2 = parent2[:common_node_index2] + parent1[common_node_index1:]
            else:
                kid1 = parent1
                kid2 = parent2

            # Mutate kids in probability
            mutate_kid_in_probability(kid1)
            mutate_kid_in_probability(kid2)

            # Append kids to new_population
            new_population.append(kid1)
            new_population.append(kid2)

        # end of generation - new_population replace the old one
        population = new_population
        population = sorted(population, key=len)
        printed = population

        # Check if min_length has changed
        if min_length == previous_min_length:
            unchanged_iterations += 1
        else:
            unchanged_iterations = 0
            previous_min_length = min_length

        generation += 1
        unchanged_iterations += 1

    return (population[0], printed)


def find_path(starting_locations, goal_locations, search_method, detail_output):
    # reading the csv file
    script_dir = os.path.dirname(__file__)
    adjacency_file_path = os.path.join(script_dir, 'adjacency.csv')
    graph = read_from_csv(adjacency_file_path)

    # turn the arrays to one long string
    starting_locations = [' ; '.join(starting_locations)]
    goal_locations = [' ; '.join(goal_locations)]

    # validation check for input
    valid_location_list(graph, starting_locations, "starting")
    valid_location_list(graph, goal_locations, "goal")

    # cutting the given starting_locations
    string_starting_locations = starting_locations[0]
    s_locations = [location.strip() for location in
                   string_starting_locations.split(';')]  # handle number of starting location

    # cutting the given goal_locations
    string_goal_locations = goal_locations[0]
    g_locations = [location.strip() for location in string_goal_locations.split(';')]  # handle number of goal location

    paths = []
    printeds = []

    for start in s_locations:
        s_parts = start.split(", ")
        s_color, s_name, s_state = s_parts[0], s_parts[1], s_parts[2]

        for goal in g_locations:
            g_parts = goal.split(", ")
            g_color, g_name, g_state = g_parts[0], g_parts[1], g_parts[2]

            if s_color == g_color:
                start = graph[s_name + ", " + s_state]
                goal = graph[g_name + ", " + g_state]

                if search_method == 1:  # A_star
                    found = A_star(graph, start, goal)
                    if found:
                        path = return_path(start, goal)
                        paths.append(path)
                    else:
                        paths.append(["No path found"])

                elif search_method == 2:  # hill climbing
                    found = hill_climbing(graph, start, goal)
                    if found:
                        path = return_path(start, goal)  # check return path - error!
                        paths.append(path)
                    else:
                        paths.append(["No path found"])

                elif search_method == 3:  # simulated annealing
                    found, printed = simulated_annealing(graph, start, goal)
                    if found:
                        path = return_path(start, goal)
                        paths.append(path)
                        printeds.append((printed))
                    else:
                        paths.append(["No path found"])
                        printeds.append(["No path found"])

                elif search_method == 4:  # k-beam
                    found, printed = k_beam(graph, start, goal)
                    if found:
                        path = return_path(start, goal)
                        paths.append(path)
                        printed = ['{}, {}, {}'.format(printed[i], printed[i+1], printed[i+2]) for i in range(0, len(printed), 3)]
                        printeds.append(printed)
                    else:
                        paths.append(["No path found"])
                        printeds.append(["No path found"])

                elif search_method == 5:  # genetic
                    found, printed = genetic_algorithm(graph, start, goal)
                    if found:
                        paths.append(found)
                        printeds.append(printed)
                    else:
                        paths.append(["No path found"])
                        printeds.append(["No path found"])

            else:
                paths.append(["No path found"])
                printeds.append(["No path found"])

    semi_final_paths = []
    final_paths = []
    final_printed = []

    # print(printeds)

    # Calculate lengths (assign 1000 where path is 'No path found') for any path in paths
    for path in paths:
        if len(path) == 1:
            semi_final_paths.append(1000)
        else:
            semi_final_paths.append(len(path))

    # Reshape the list into a square matrix for cost_matrix
    n = int(len(semi_final_paths) ** 0.5)
    cost_matrix = np.array(semi_final_paths).reshape(n, n)

    # Solve the assignment problem using the Hungarian Algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # append to final_path the min paths
    for i in row_ind:
        final_paths.append(paths[row_ind[i] * n + col_ind[i]])
        if search_method == 3:
            final_printed.append(printeds[row_ind[i] * n + col_ind[i]])
        if search_method == 4:
            final_printed.append(printeds[row_ind[i] * n + col_ind[i]])
        if search_method == 5:
            final_printed.append(printeds[row_ind[i] * n + col_ind[i]])

    # return the max lenght of path
    lenght = 0
    for i in final_paths:
        if len(i) > lenght:
            lenght = len(i)

    # form of output
    for i in range(lenght):
        print_line = "{"
        print_True = "Heuristic: {"
        for j in range(len(s_locations)):
            if i == 2 and detail_output == True and search_method == 1:
                if (final_paths[j][0] == "No path found"):
                    print_True += "No path found ; "
                else:
                    _, state1 = final_paths[j][i - 1].split(", ")
                    state1 = state1[0:2]
                    _, state2 = final_paths[j][-1].split(", ")
                    state2 = state2[0:2]
                    print_True += str(heuristic_calculation(state1, state2)) + " ; "

            try:
                print_line += final_paths[j][i] + " ; "
            except IndexError:
                print_line += final_paths[j][-1] + " ; "

        if i == 2 and detail_output == True and search_method == 4:
            for item in final_printed:
                    print("\nbag: {" + f"{{{item}}}"[3:-3] + '}')
            print("")

        if i == 2 and detail_output == True and search_method == 3:
            for item in final_printed:
                print("action: {" + f"{{{item}}}"[1:-1] + '}')
            print("")

        if i == 2 and detail_output == True and search_method == 5:
            for item in final_printed:
                print("\n population: {" + f"{{{item}}}"[3:-3] + '}')
            print("")

        if i == 2 and detail_output == True and search_method == 1:
            print_True += "}"
            print_True = print_True[0:-4] + '}'
            print(print_True)

        print_line = print_line[0:-3]
        print_line += "}"
        print(print_line)


