from itertools import combinations
from time import time, sleep
import argparse
import random
import numpy as np
import os
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


def brute_force_partition(G, k):
    # Compute the number of nodes on each side of the partition
    n = len(G) // 2
    nodes = list(G.keys())
    # Generate all possible subsets of size n using itertools
    for U in combinations(nodes, n):
        # Compute the complement set W
        W = set(nodes) - set(U)
        # Count the number of edges between U and W
        edge_count = count_edges_between(U, W, G)
        # Check if the count is less than or equal to k
        if edge_count <= k:
            # If so, return True, indicating that a partition of size k or less exists
            return True
    # If no partition of size k or less exists, return False
    return False


def kernighan_lin_partition(G, n, k):
    nodes = np.array(list(G.keys()))
    np.random.shuffle(nodes)  # randomize the order of nodes
    U, W = np.split(nodes, 2)  # split the nodes into two RANDOM sets

    while True:
        costs = {}
        for u in U:
            for w in W:
                costU = sum([v in W for v in G[u]]) - \
                    sum([v in U for v in G[u]])
                costW = sum([v in U for v in G[w]]) - \
                    sum([v in W for v in G[w]])
                costs[(u, w)] = costU + costW - 2 * (u in G[w])

        # Finding the pair with the minimum cost i.e. largest gain
        minCostPair = min(costs, key=costs.get)
        U[U == minCostPair[0]] = minCostPair[1]  # Swapping the nodes
        W[W == minCostPair[1]] = minCostPair[0]  # Swapping the nodes

        if sum([w in G[u] for u in U for w in W]) <= k:
            return True  # if the number of edges between U and W is less than or equal to k, return True
        elif (min(costs.values()) >= 0):
            return False  # if the minimum cost is greater than or equal to 0 i.e. no more gain, return False


def count_edges_between(U, W, G):
    # Count the number of edges between the two sets
    edge_count = 0
    for u in U:
        for w in W:
            if w in G[u]:
                edge_count += 1
    return edge_count


def generate_random_graph(n):
    # Create an empty graph
    G = {}
    # Add all n nodes to the graph
    for i in range(n):
        G[i] = set()
    # For each pair of nodes, add an edge with probability 1/2
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < 0.5:
                G[i].add(j)
                G[j].add(i)
    return G


# def test_brute_force_partition(n, k, num_tests=10):
#     G = generate_random_graph(n)
#     total_execution_time = 0
#     for i in range(num_tests):
#         start_time = time()
#         result = brute_force_partition(G, k)
#         end_time = time()
#         execution_time = end_time - start_time
#         total_execution_time += execution_time

#     average_execution_time = total_execution_time / num_tests
#     return result, average_execution_time


test_no = 1
different_count = 0
bruteTimeMeasurements = []
heuristicTimeMeasurements = []


def compareAlgorithms(n, testAmount):
    global test_no
    global different_count
    global bruteTimeMeasurements
    global heuristicTimeMeasurements
    with open("BruteForceAlgorithmResults.txt", "a") as bf_file, open("HeuristicAlgorithmResults.txt", "a") as h_file, open("Comparison.txt", "a") as c_file:
        bruteTotalTime = 0
        heuristicTotalTime = 0
        bruteTrueCount = 0
        heuristicTrueCount = 0
        for k in range(1, ((n*(n-1)//2))+1):
            for i in range(testAmount):
                G = generate_random_graph(n)  # Generate a random graph

                # Test on the brute force algorithm
                start_time = time()
                bruteTestResult = brute_force_partition(G, k)
                end_time = time()
                bruteTestTime = end_time - start_time
                bruteTimeMeasurements.append(bruteTestTime)

                bf_file.write(
                    f"Test No: {test_no}\t\t\t\tPartition exists: {bruteTestResult}\n")
                if bruteTestResult:
                    bruteTrueCount += 1

                # Test on the heuristic algorithm
                start_time = time()
                heuristicTestResult = kernighan_lin_partition(G, n, k)
                end_time = time()
                heuristicTestTime = end_time - start_time
                heuristicTimeMeasurements.append(heuristicTestTime)

                h_file.write(
                    f"Test No: {test_no}\t\t\t\tPartition exists: {heuristicTestResult}\n")
                if heuristicTestResult:
                    heuristicTrueCount += 1

                c_file.write(
                    f"Test No: {test_no}\t\t\t\tResults are : {'same' if bruteTestResult == heuristicTestResult else 'different'}\n")

                if (bruteTestResult != heuristicTestResult):
                    different_count += 1

                bruteTotalTime += bruteTestTime
                heuristicTotalTime += heuristicTestTime
                test_no += 1

        print("Test completed and results written to files.")
        return bruteTotalTime, heuristicTotalTime, bruteTrueCount, heuristicTrueCount


def printGraph(G):
    for node, edges in G.items():
        print(f'Node {node} has edges with: {", ".join(map(str, edges))}')


def makeFunctionalityTest():
    # Black box testing
    # Test 1
    G = {
        0: {1, 2},
        1: {0, 2},
        2: {0, 1},
        3: {4, 5},
        4: {3, 5},
        5: {3, 4}
    }
    k = 0
    print("-" * 20)
    print("Test 1: disjoint graph")
    print("Result should be True")
    print("Brute Force Algorithm: ", brute_force_partition(G, k))
    print("Heuristic Algorithm: ", kernighan_lin_partition(G, 6, k))
    print("-" * 20)

    # Test 2
    G = {
        0: {1, 2, 3},
        1: {0, 2, 3},
        2: {0, 1, 3},
        3: {0, 1, 2},
    }
    k = 1
    print("-" * 20)
    print("Test 2")
    print("Result should be False")
    print("Brute Force Algorithm: ", brute_force_partition(G, k))
    print("Heuristic Algorithm: ", kernighan_lin_partition(G, 4, k))
    print("-" * 20)

    # Test 3
    G = {
        0: {1, 2},
        1: {0},
        2: {0},
        3: {4, 0},
        4: {3, 0, 5},
        5: {4}
    }
    k = 1
    print("-" * 20)
    print("Test 3")
    print("Result should be True")
    print("Brute Force Algorithm: ", brute_force_partition(G, k))
    print("Heuristic Algorithm: ", kernighan_lin_partition(G, 6, k))
    print("-" * 20)


if __name__ == "__main__":
    if os.path.exists("Results.txt"):
        os.remove("Results.txt")
    if os.path.exists("BruteForceAlgorithmResults.txt"):
        os.remove("BruteForceAlgorithmResults.txt")
    if os.path.exists("HeuristicAlgorithmResults.txt"):
        os.remove("HeuristicAlgorithmResults.txt")
    if os.path.exists("Comparison.txt"):
        os.remove("Comparison.txt")

    parser = argparse.ArgumentParser(description='Brute Force Partition')
    parser.add_argument('--nMin', type=int, default=10,
                        help='Number of nodes (MIN)')
    parser.add_argument('--nMax', type=int, default=10,
                        help='Number of nodes (MAX)')
    parser.add_argument('--num_tests', type=int, default=3,
                        help='Number of tests to run')

    args = parser.parse_args()
    nMin = args.nMin
    nMax = args.nMax
    num_tests = args.num_tests
    old_test_no = 0

    makeFunctionalityTest()

    numNodesList = []
    bruteAvgTimeList = []
    heuristicAvgTimeList = []
    for n in range(nMin, nMax+1, 2):
        brTotal, heuTotal, brtTrue, heuTrue = compareAlgorithms(n, num_tests)
        totalTests = num_tests * (n*(n-1)//2)
        #totalTests = num_tests
        numNodesList.append(n)
        bruteAvgTimeList.append(brTotal/totalTests)
        heuristicAvgTimeList.append(heuTotal/totalTests)

        with open("Results.txt", "a") as file:
            # file.write(
            #     f"Number of nodes: {n}\t\tBrute Force TOTAL Time: {brTotal}\t\tHeuristic TOTAL Time: {heuTotal}\t\tBruteTOTAL/HeuristicTOTAL: {brTotal/heuTotal}\t Brute True Count: {brtTrue}\t Heuristic True Count: {heuTrue}\n"
            # )
            file.write("\n\n")
            file.write(f"Number of nodes: {n}\n")
            file.write(f"Number of tests: {totalTests}\n")
            file.write(f"Brute Force TOTAL Time (seconds): {brTotal}\n")
            file.write(f"Heuristic TOTAL Time (seconds): {heuTotal}\n")
            file.write(
                f"BruteTOTAL/HeuristicTOTAL (seconds): {brTotal/heuTotal}\n")
            file.write(f"Brute True Count: {brtTrue}\n")
            file.write(f"Heuristic True Count: {heuTrue}\n")
            file.write(
                f"Heuristic Algorithm Accuracy: {1 - (different_count/totalTests) }\n")
            file.write("*" * 15 + "\n")
            file.write("\n\n")

    if (nMin == nMax):
        bruteMeasurementsMean = np.mean(bruteTimeMeasurements)
        heuristicMeasurementsMean = np.mean(heuristicTimeMeasurements)

        bruteMeasurementsStd = np.std(bruteTimeMeasurements)
        heuristicMeasurementsStd = np.std(heuristicTimeMeasurements)

        bruteMeasurementsStdErr = stats.sem(bruteTimeMeasurements)
        heuristicMeasurementsStdErr = stats.sem(heuristicTimeMeasurements)

        bruteConfidenceInterval = stats.norm.interval(
            0.90, loc=bruteMeasurementsMean, scale=bruteMeasurementsStdErr)

        heuristicConfidenceInterval = stats.norm.interval(
            0.90, loc=heuristicMeasurementsMean, scale=heuristicMeasurementsStdErr)

        # Interval [a-b, a+b] is narrow enough if b/a < 0.1
        a = (bruteConfidenceInterval[0] + bruteConfidenceInterval[1]) / 2
        b = (bruteConfidenceInterval[1] - bruteConfidenceInterval[0]) / 2
        if (b/a < 0.1):
            bruteConfidenceInterval = (a-b, a+b)

        a = (heuristicConfidenceInterval[0] +
             heuristicConfidenceInterval[1]) / 2
        b = (heuristicConfidenceInterval[1] -
             heuristicConfidenceInterval[0]) / 2
        if (b/a < 0.1):
            heuristicConfidenceInterval = (a-b, a+b)
        print("\n\n")
        print("Brute Force running time measurements:")
        print(f"Mean: {bruteMeasurementsMean}")
        print(f"Interval with 90% confidence: {bruteConfidenceInterval}")
        print("\n\n")
        print("\nHeuristic running time measurements:")
        print(f"Mean: {heuristicMeasurementsMean}")
        print(f"Interval with 90% confidence: {heuristicConfidenceInterval}")
        print("\n\n")

    # converting list to numpy array for calculations
    node_sizes = np.array(numNodesList)
    brute_times = np.array(bruteAvgTimeList)
    heuristic_times = np.array(heuristicAvgTimeList)

    # Fit a line to the Brute Force data
    brute_coeffs = np.polyfit(np.log(node_sizes), np.log(brute_times), 1)
    brute_polynomial = np.poly1d(brute_coeffs)

    # Fit a line to the Heuristic data
    heuristic_coeffs = np.polyfit(
        np.log(node_sizes), np.log(heuristic_times), 1)
    heuristic_polynomial = np.poly1d(heuristic_coeffs)

    # Print the equations of the fitted lines
    brute_a = np.exp(brute_polynomial[1])
    brute_b = brute_polynomial[0]
    print(
        f'The equation of the fitted line for Brute Force is: y = {brute_a}x^{brute_b}')

    heuristic_a = np.exp(heuristic_polynomial[1])
    heuristic_b = heuristic_polynomial[0]
    print(
        f'The equation of the fitted line for Heuristic is: y = {heuristic_a}x^{heuristic_b}')

    # Generate y-values for the fitted lines
    brute_y_fit = np.exp(brute_polynomial(np.log(node_sizes)))
    heuristic_y_fit = np.exp(heuristic_polynomial(np.log(node_sizes)))

    # Plot the original data and the fitted lines
    plt.figure()
    plt.plot(node_sizes, brute_times, 'ko', label='Brute Force Data')
    plt.plot(node_sizes, brute_y_fit, 'r-', label='Brute Force Fitted Line')
    plt.plot(node_sizes, heuristic_times, 'bo', label='Heuristic Data')
    plt.plot(node_sizes, heuristic_y_fit, 'g-', label='Heuristic Fitted Line')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of Nodes (log scale)')
    plt.ylabel('Total Time (log scale)')
    plt.title('Time Complexity')
    plt.grid(True)
    plt.show()
