#!/usr/bin/python3
from copy import copy, deepcopy

from which_pyqt import PYQT_VER

if PYQT_VER == 'PYQT5':
    from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
    from PyQt4.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT6':
    from PyQt6.QtCore import QLineF, QPointF
else:
    raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))

import time
import numpy as np
from TSPClasses import *
import heapq
import itertools


class QueuedItems:
    def __init__(self, costMatrix, current_depth, current_cost, column_to_search, current_path):
        self.costMatrix = costMatrix
        self.current_depth = current_depth
        self.current_cost = current_cost
        self.current_column = column_to_search
        self.current_path = current_path


created = 0
pruned = 0
solution = 0
updated_solution = 0
max_queue_size = 0
def incUpdateSolution():
    global updated_solution
    updated_solution += 1

def incCreated():
    global created
    created += 1
    pass

def incPruned():
    global pruned
    pruned += 1
    pass

def findPathAndPopulateQueue(lowerBound, bssf_value, costMatrix, priority_queue, current_column, ncities, current_path,
                             current_depth, list_of_cities):
    # O(n^2) we run through this n^2 times and all operations are O(1)
    for i in range(ncities):
        if costMatrix[current_column][i] != np.inf:
            cost = lowerBound + costMatrix[current_column][i]
            if cost < bssf_value:
                current_path.append(i)
                temp_cost_matrix = deepcopy(costMatrix)
                for j in range(ncities):
                    temp_cost_matrix[current_column][j] = np.inf
                    temp_cost_matrix[j][i] = np.inf
                if i != 0 or abs(current_depth) == ncities - 1:
                    temp_path = deepcopy(current_path)
                    incCreated()
                    heapq.heappush(priority_queue, (current_depth - 1, cost, temp_cost_matrix, temp_path, i))
                else:
                    incPruned()
                current_path.pop(len(current_path) - 1)
            else:
                incPruned()
            # print(temp_cost_matrix)
            # print(cost)
            # print("These are our costs and matrices")
    # print(priority_queue)

    pass


def tryDifferentRoutes(current_solution, route):
    better_solutions = []
    if type(route) == TSPSolution:
       length = len(route.route)
    else:
        length = len(route)
    best_solution = deepcopy(current_solution)
    for i in range(length):
        for j in range(i, length):
            route_copy = route if type(route) != TSPSolution else route.route
            route_copy[i], route_copy[j] = route_copy[j], route_copy[i]
            possible_solution = TSPSolution(route_copy)
            #if possible_solution.cost < current_solution['cost']:
             #   print("Our possible solution costs " + str(possible_solution.cost))
              #  print("Our old solution had a route cost of " + str(current_solution['cost']))
            if best_solution['cost'] > possible_solution.cost:
                    best_solution['cost'] = deepcopy(possible_solution.cost)
                    best_solution['soln'] = deepcopy(possible_solution.route)
            route_copy[i], route_copy[j] = route_copy[j], route_copy[i]



               # better_solutions.append(possible_solution)
    return [best_solution]
    pass


class TSPSolver:
    def __init__(self, gui_view):
        self._scenario = None

    def setupWithScenario(self, scenario):
        self._scenario = scenario

    ''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution,
		time spent to find solution, number of permutations tried during search, the
		solution found, and three null values for fields not used for this
		algorithm</returns>
	'''

    def defaultRandomTour(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        bssf = None
        start_time = time.time()
        while not foundTour and time.time() - start_time < time_allowance:
            # create a random permutation
            perm = np.random.permutation(ncities)
            route = []
            # Now build the route using the random permutation
            for i in range(ncities):
                route.append(cities[perm[i]])
            bssf = TSPSolution(route)
            count += 1
            if bssf.cost < np.inf:
                # Found a valid route
                foundTour = True
        end_time = time.time()
        results['cost'] = bssf.cost if foundTour else math.inf
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    ''' <summary>
		This is the entry point for the greedy solver, which you must implement for
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this
		algorithm</returns>
	'''

    def greedy(self, time_allowance=60.0, starting_column = 0):
        cities = self._scenario.getCities()
        ncities = len(cities)
        start_time = time.time()
        original_column = starting_column
        costMatrix = [[0 for i in range(ncities)] for j in range(ncities)]
        path = []
        path.append(starting_column)
        for i in range(ncities):
            for j in range(ncities):
                costMatrix[i][j] = cities[i].costTo(cities[j])
        for i in range(ncities):
            min_value = np.inf
            current_index = -1
            for j in range(ncities):
                current_value = costMatrix[starting_column][j]
                if current_value < min_value and j != original_column:
                    min_value = current_value
                    current_index = j
            path.append(current_index)
            starting_column = current_index
            for j in range(ncities):
                costMatrix[j][starting_column] = np.inf
        city_path =[]
        results = {}
        for i in range(len(path) - 1):
            city_path.append(cities[path[i]])
        route = TSPSolution(city_path)
        end_time = time.time()
       # print("The cost is " + str(route.cost))
        results['cost'] = route.cost
        results['time'] = end_time - start_time
        results['count'] = None
        results['soln'] = route
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
      #  print(path)
        return results

        pass

    ''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints:
		max queue size, total number of states created, and number of pruned states.</returns>
	'''

    def branchAndBound(self, time_allowance=60.0):
        cities = self._scenario.getCities()
        ncities = len(cities)
        global pruned
        pruned = 0
        global created
        created = 0
        global updated_solution
        updated_solution = 0
        global max_queue_size
        max_queue_size = 0
        smallest_solution = self.greedy(5.0, 0)
        for i in range(1, ncities):
            possible_solution = self.greedy(5.0, i)
            if possible_solution['cost'] < smallest_solution['cost']:
                smallest_solution = possible_solution
                print("Our new solution has a cost of " + str(possible_solution['cost']))
        bssf = smallest_solution
        results = bssf
        route = bssf['soln']
        print("our initial bssf cost is " + str(bssf['cost']))

        start_time = time.time()
        costMatrix = [[0 for i in range(ncities)] for j in range(ncities)]
        # this is O(n^2) because that is what it takes to create the cost matrix
        # because we calculate the value of traveling from each city to the others
        # The space complexity of this part is O(n^2) because that is the space taken up by our cost matrix
        for i in range(ncities):
            for j in range(ncities):
                costMatrix[i][j] = cities[i].costTo(cities[j])
        print(costMatrix)
        lowerBound = 0
        # O(n^2) as well because we must visit each value in the matrix to create our min cost matrix
        for i in range(ncities):
            # This is O(n) because it looks at each value in the list to determine the min
            localMin = min(costMatrix[i])
            # O(n) because we visit each value in the matrix and change it is as necessary
            for j in range(ncities):
                if (costMatrix != np.inf):
                    costMatrix[i][j] = costMatrix[i][j] - localMin
            lowerBound = lowerBound + localMin
        # O(n^2) It works exactly like the one above but it makes sure each column has a zero in it rather than just each row like above
        for i in range(ncities):
            column_min = np.inf
            for j in range(ncities):
                if column_min > costMatrix[j][i]:
                    column_min = costMatrix[j][i]
            for j in range(ncities):
                costMatrix[j][i] = costMatrix[j][i] - column_min
            lowerBound = lowerBound + column_min
        priority_queue = []
        findPathAndPopulateQueue(lowerBound, bssf['cost'], costMatrix, priority_queue, 0, ncities, [0], 0, cities)
        results = {}
        results = bssf
        final_path = []
        # This entire while loop can have complexity of up to O(n!) if we were to have to visit every single possible
        # path because it calls a function n^2 it has O(n^2*n!) worst case
        # The space complexity also is O(n!&n^2) because that is the most that matrices that could be stored in our priority queue
        while len(priority_queue) > 0:
            max_queue_size = len(priority_queue) if len(priority_queue) > max_queue_size else max_queue_size
            # print("Our priority queue has length " + str(len(priority_queue)))
            depth, cost, matrix, path, column = heapq.heappop(priority_queue)
            while cost >= bssf['cost'] and len(priority_queue) > 0:
                depth, cost, matrix, path, column = heapq.heappop(priority_queue)
            if abs(depth) == ncities:
                print("Our temp path is " + str(path))
                final_path = deepcopy(path)
                city_path = []
                for i in range(len(path) - 1):
                    city_path.append(cities[path[i]])
                route = TSPSolution(city_path)
                end_time = time.time()
                incUpdateSolution()
                bssf['cost'] = route.cost
                print("The cost is " + str(route.cost))
                results['cost'] = route.cost
                results['time'] = end_time - start_time
                results['count'] = updated_solution
                results['soln'] = route
                results['max'] = max_queue_size
                results['total'] = created
                results['pruned'] = pruned
            else:
                findPathAndPopulateQueue(cost, bssf['cost'], matrix, priority_queue, column, ncities, path, depth,
                                         cities)
            if time.time() - start_time >= time_allowance:
                end_time = time.time()
                results['cost'] = route.cost
                results['time'] = time_allowance
                results['count'] = updated_solution
                results['max'] = max_queue_size
                results['total'] = created
                results['pruned'] = pruned + len(priority_queue)
                return results
        print("In the end our this is our path ")
        print(final_path)
        end_time = time.time()
        results['time'] = end_time - start_time
        results['max'] = max_queue_size
        results['total'] = created
        results['pruned'] = pruned
        return results
        pass

    ''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number of solutions found during search, the
		best solution found.  You may use the other three field however you like.
		algorithm</returns>
	'''

    def fancy(self, time_allowance=60.0):
        cities = self._scenario.getCities()
        ncities = len(cities)
        start_time = time.time()
        smallest_solution = self.greedy(5.0, 0)
        second_smallest = self.greedy(5.0, 0)
        third_smallest = self.greedy(5.0, 0)
        for i in range(1, ncities):
            possible_solution = self.greedy(5.0, i)
            if possible_solution['cost'] < smallest_solution['cost']:
                third_smallest = second_smallest
                second_smallest = smallest_solution
                smallest_solution = possible_solution
                print("Our new solution has a cost of " + str(possible_solution['cost']))
            elif possible_solution['cost'] < second_smallest['cost']:
                third_smallest = second_smallest
                second_smallest = possible_solution
            elif possible_solution['cost'] < third_smallest['cost']:
                third_smallest = possible_solution
        route = smallest_solution['soln'].route
        route_list = tryDifferentRoutes(smallest_solution, route)
        print("Our smallest solution is " + str(smallest_solution['cost']))
        while len(route_list) > 0:
            attempt_route = route_list.pop()
            other_solution = smallest_solution
            other_solution['cost'] = attempt_route['cost']
            other_solution['soln'] = attempt_route['soln']
            temp_list = tryDifferentRoutes(other_solution, attempt_route['soln'])
            for item in temp_list:
                if item['cost'] != other_solution['cost']:
                    route_list.append(item)

            print("Our route length is " + str(len(route_list)))
        print(route)

        end_time = time.time()
        print("Our smallest solution is " + str(smallest_solution['cost']))
        print("Our second smallest solution is " + str(second_smallest['cost']))
        print("Our third smallest solution is " + str(third_smallest['cost']))
        smallest_solution['time'] = end_time - start_time

        print(smallest_solution['soln'])
        smallest_solution['soln'] = TSPSolution(smallest_solution['soln'])
        return smallest_solution

        pass
