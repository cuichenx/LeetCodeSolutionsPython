from collections import defaultdict
from typing import List

from helpers import print_assert
import heapq
import numpy as np

class Solution:
    def minMovesToSeat(self, seats: List[int], students: List[int]) -> int:
        seats.sort()
        students.sort()
        c = 0
        for seat, student in zip(seats, students):
            c += abs(seat-student)

        return c

    def test1(self):
        print_assert(self.minMovesToSeat([3, 1, 5], [2, 7, 4]), 4)
        print_assert(self.minMovesToSeat([4, 1, 5, 9], [1, 3, 2, 6]), 7)

    def winnerOfGame(self, colors: str) -> bool:
        A_moves = 0
        B_moves = 0

        cur_A_streak = 0
        cur_B_streak = 0
        for c in colors:
            if c == 'A':
                cur_A_streak += 1
                cur_B_streak = 0
                if cur_A_streak >=3 :
                    A_moves += 1
            elif c == 'B':
                cur_B_streak += 1
                cur_A_streak = 0
                if cur_B_streak >= 3:
                    B_moves += 1

        return A_moves > B_moves

    def test2(self):
        print_assert(self.winnerOfGame("AAABABB"), True)
        print_assert(self.winnerOfGame("AA"), False)
        print_assert(self.winnerOfGame("ABBBBBBBAAA"), False)
        print_assert(self.winnerOfGame("AAABBB"), False)
        print_assert(self.winnerOfGame("AAAABBB"), True)
        print_assert(self.winnerOfGame("AAAABBBB"), False)
        print_assert(self.winnerOfGame("AAAAABBBB"), True)
        print_assert(self.winnerOfGame("A"), False)
        print_assert(self.winnerOfGame("B"), False)
        print_assert(self.winnerOfGame("BB"), False)
        print_assert(self.winnerOfGame("BBB"), False)

    def networkBecomesIdle(self, edges: List[List[int]], patience: List[int]) -> int:
        n = len(patience)
        distances = self.dijkstra(set(tuple(e) for e in edges), n, 0)
        max_wait = 0
        for dist, pat in zip(distances[1:], patience[1:]):
            last_msg_time = (dist*2-1)//pat*pat
            max_wait = max(max_wait, dist*2+1+last_msg_time)
        return max_wait

class Heap():

    def __init__(self):
        self.array = []
        self.size = 0
        self.pos = []

    def newMinHeapNode(self, v, dist):
        minHeapNode = [v, dist]
        return minHeapNode

    # A utility function to swap two nodes
    # of min heap. Needed for min heapify
    def swapMinHeapNode(self, a, b):
        t = self.array[a]
        self.array[a] = self.array[b]
        self.array[b] = t

    # A standard function to heapify at given idx
    # This function also updates position of nodes
    # when they are swapped.Position is needed
    # for decreaseKey()
    def minHeapify(self, idx):
        smallest = idx
        left = 2 * idx + 1
        right = 2 * idx + 2

        if left < self.size and self.array[left][1] < self.array[smallest][1]:
            smallest = left

        if right < self.size and self.array[right][1] < self.array[smallest][1]:
            smallest = right

        # The nodes to be swapped in min
        # heap if idx is not smallest
        if smallest != idx:
            # Swap positions
            self.pos[self.array[smallest][0]] = idx
        self.pos[self.array[idx][0]] = smallest


        # Swap nodes
        self.swapMinHeapNode(smallest, idx)

        self.minHeapify(smallest)


    # Standard function to extract minimum
    # node from heap
    def extractMin(self):
        # Return NULL wif heap is empty
        if self.isEmpty() == True:
            return

        # Store the root node
        root = self.array[0]

        # Replace root node with last node
        lastNode = self.array[self.size - 1]
        self.array[0] = lastNode

        # Update position of last node
        self.pos[lastNode[0]] = 0
        self.pos[root[0]] = self.size - 1

        # Reduce heap size and heapify root
        self.size -= 1
        self.minHeapify(0)

        return root


    def isEmpty(self):
        return True if self.size == 0 else False


    def decreaseKey(self, v, dist):
        # Get the index of v in  heap array

        i = self.pos[v]

        # Get the node and update its dist value
        self.array[i][1] = dist

        # Travel up while the complete tree is
        # not hepified. This is a O(Logn) loop
        while i > 0 and self.array[i][1] < self.array[(i - 1) / 2][1]:
            # Swap this node with its parent
            self.pos[self.array[i][0]] = (i - 1) / 2
            self.pos[self.array[(i - 1) / 2][0]] = i
            self.swapMinHeapNode(i, (i - 1) / 2)

            # move to parent index
            i = (i - 1) / 2


    # A utility function to check if a given
    # vertex 'v' is in min heap or not
    def isInMinHeap(self, v):
        if self.pos[v] < self.size:
            return True
        return False


def printArr(dist, n):
    print ("Vertex\tDistance from source")
    for i in range(n):
        print("%d\t\t%d" % (i, dist[i]))


class Graph():

    def __init__(self, V):
        self.V = V
        self.graph = defaultdict(list)

    # Adds an edge to an undirected graph
    def addEdge(self, src, dest, weight):

        # Add an edge from src to dest.  A new node
        # is added to the adjacency list of src. The
        # node is added at the beginning. The first
        # element of the node has the destination
        # and the second elements has the weight
        newNode = [dest, weight]
        self.graph[src].insert(0, newNode)

        # Since graph is undirected, add an edge
        # from dest to src also
        newNode = [src, weight]
        self.graph[dest].insert(0, newNode)

    # The main function that calculates distances
    # of shortest paths from src to all vertices.
    # It is a O(ELogV) function
    def dijkstra(self, src):

        V = self.V  # Get the number of vertices in graph
        dist = []  # dist values used to pick minimum
        # weight edge in cut

        # minHeap represents set E
        minHeap = Heap()

        #  Initialize min heap with all vertices.
        # dist value of all vertices
        for v in range(V):
            dist.append(self.V+1)
            minHeap.array.append(minHeap.
                                 newMinHeapNode(v, dist[v]))
            minHeap.pos.append(v)

        # Make dist value of src vertex as 0 so
        # that it is extracted first
        minHeap.pos[src] = src
        dist[src] = 0
        minHeap.decreaseKey(src, dist[src])

        # Initially size of min heap is equal to V
        minHeap.size = V

        # In the following loop,
        # min heap contains all nodes
        # whose shortest distance is not yet finalized.
        while minHeap.isEmpty() == False:

            # Extract the vertex
            # with minimum distance value
            newHeapNode = minHeap.extractMin()
            u = newHeapNode[0]

            # Traverse through all adjacent vertices of
            # u (the extracted vertex) and update their
            # distance values
            for pCrawl in self.graph[u]:

                v = pCrawl[0]

                # If shortest distance to v is not finalized
                # yet, and distance to v through u is less
                # than its previously calculated distance
                if minHeap.isInMinHeap(v) and dist[u] != sys.maxint and pCrawl[1] + dist[u] < dist[v]:
                    dist[v] = pCrawl[1] + dist[u]

                    # update distance value
                    # in min heap also
                    minHeap.decreaseKey(v, dist[v])

        printArr(dist, V)

    # def dijkstra(self, edges: set, num_vertices: int, src: int):
    #     # https://www.geeksforgeeks.org/dijkstras-shortest-path-algorithm-greedy-algo-7/
    #     dist_heap = [(num_vertices+999, i) for i in range(num_vertices)]
    #     dist_heap[0] = 0
    #     heapq.heapify(dist_heap)
    #     dist_out = [-1] * num_vertices
    #     sptSet = [False] * num_vertices
    #
    #     def minDistance():
    #         min_dist, min_index = heapq.heappop(dist_heap)
    #         dist_out[min_index] = min_dist
    #         return min_index
    #
    #     for cout in range(num_vertices):
    #         # Pick the minimum distance vertex from
    #         # the set of vertices not yet processed.
    #         # x is always equal to src in first iteration
    #         x = minDistance()
    #         # Put the minimum distance vertex in the
    #         # shortest path tree
    #         sptSet[x] = True
    #         # Update dist value of the adjacent vertices
    #         # of the picked vertex only if the current
    #         # distance is greater than new distance and
    #         # the vertex in not in the shortest path tree
    #         for y in range(num_vertices):
    #             if ((x, y) in edges or (y, x) in edges) and sptSet[y] == False and dist_out[y] > dist_out[x] + 1:
    #                 dist_out[y] = dist_out[x] + 1
    #
    #     return dist_out



if __name__ == '__main__':
    sol = Solution()
    sol.test3()

