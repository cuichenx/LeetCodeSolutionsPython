import sys
from collections import deque
from typing import List, Optional

sys.path.append("..")
from helpers import print_assert, print_matrix, ListNode, LinkedList as ll
import copy
import time

FREE = 0
ATTACKABLE = 1
OCCUPIED = 2
class Q52:
    # 52. N-Queens II
    # Given an integer n, return the number of distinct solutions to the n-queens puzzle.

    def totalNQueens(self, n: int) -> int:
        self.n = n
        # 0 means available, 1 means occupied or attackable
        empty_board = [[FREE for _ in range(n)] for _ in range(n)]
        return sum(self.numQueensPartialBoard(0, j, copy.deepcopy(empty_board)) for j in range(n))  # can only search half the board

    def numQueensPartialBoard(self, i, j, board):
        board[i][j] = OCCUPIED
        if i == self.n - 1:
            # last row
            # print_matrix(board)
            # print('-' * 10)
            return 1

        # set new attackable cells
        self.setAttackableCells(i, j, board)

        # search for each row beneath it
        num_sol = 0
        for next_j in range(self.n):
            if board[i+1][next_j] == FREE:
                num_sol += self.numQueensPartialBoard(i+1, next_j, copy.deepcopy(board))
        return num_sol


    def setAttackableCells(self, i, j, board):
        # because we're searching row by row, there's no need to set the row as ATTACKABLE.
        # this column
        for col in range(i+1, self.n):
            board[col][j] = ATTACKABLE

        # this diagonal
        r, c = i+1, j+1
        while self.inRange(r, c):
            board[r][c] = ATTACKABLE
            r += 1
            c += 1

        # this anti diagonal
        r, c = i+1, j-1
        while self.inRange(r, c):
            board[r][c] = ATTACKABLE
            r += 1
            c -= 1

    def inRange(self, r, c):
        return (0 <= r < self.n) and (0 <= c < self.n)

    def test(self):
        print_assert(self.totalNQueens(1), 1)
        print_assert(self.totalNQueens(2), 0)
        print_assert(self.totalNQueens(3), 0)
        print_assert(self.totalNQueens(4), 2)
        print_assert(self.totalNQueens(5), 10)
        t0 = time.time()
        self.totalNQueens(9)
        print(time.time() - t0, 'seconds')

class Q51:
    # 52. N-Queens
    # Given an integer n, return all distinct solutions to the n-queens puzzle

    def solveNQueens(self, n: int) -> List[List[str]]:
        self.n = n
        # 0 means available, 1 means occupied or attackable
        empty_board = [[FREE for _ in range(n)] for _ in range(n)]
        self.solutions = []
        for j in range(n):
            self.solveNQueensPartialBoard(0, j, copy.deepcopy(empty_board))
        return self.solutions

    def solveNQueensPartialBoard(self, i, j, board):
        board[i][j] = OCCUPIED
        if i == self.n - 1:
            # last row
            # print_matrix(board)
            # print('-' * 10)
            self.solutions.append(self.solutionify(board))
            return

        # set new attackable cells
        self.setAttackableCells(i, j, board)

        # search for each row beneath it
        solutions = []
        for next_j in range(self.n):
            if board[i+1][next_j] == FREE:
                self.solveNQueensPartialBoard(i+1, next_j, copy.deepcopy(board))


    def setAttackableCells(self, i, j, board):
        # because we're searching row by row, there's no need to set the row as ATTACKABLE.
        # this column
        for col in range(i+1, self.n):
            board[col][j] = ATTACKABLE

        # this diagonal
        r, c = i+1, j+1
        while self.inRange(r, c):
            board[r][c] = ATTACKABLE
            r += 1
            c += 1

        # this anti diagonal
        r, c = i+1, j-1
        while self.inRange(r, c):
            board[r][c] = ATTACKABLE
            r += 1
            c -= 1

    def inRange(self, r, c):
        return (0 <= r < self.n) and (0 <= c < self.n)

    def solutionify(self, board):
        out = []
        for r in board:
            s = ''
            for c in r:
                s += 'Q' if c == OCCUPIED else '.'
            out.append(s)
        return out

    def test(self):
        print(self.solveNQueens(1))
        print(self.solveNQueens(2))
        print(self.solveNQueens(3))
        print(self.solveNQueens(4))
        print(self.solveNQueens(5))


class Q160:
    # Given the heads of two singly linked-lists headA and headB, return the node at which the two lists intersect.
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        # p1: traverse A, C, B
        # p2: traverse B, C, A
        # when the two pointers match, that's the intersection
        # if both pointers are end, then no intersection
        p1 = headA
        p2 = headB
        while not (p1 is None and p2 is None):

            if p1 is None:
                p1 = headB
            if p2 is None:
                p2 = headA
            if p1 == p2:
                return p1

            p1 = p1.next
            p2 = p2.next
        return None

    def test(self):
        l1_head = ll.makeLinkedList([1, 9, 1, 2, 4])
        l2_head = ListNode(val=3, next=l1_head.next.next.next)
        print_assert(self.getIntersectionNode(l1_head, l2_head), l1_head.next.next.next)

        l1_head = ll.makeLinkedList([2, 6, 4])
        l2_head = ll.makeLinkedList([1, 5])
        print_assert(self.getIntersectionNode(l1_head, l2_head), None)

class Q88:
    # Merge nums1 and nums2 into a single array sorted in non-decreasing order.
    # The final sorted array should not be returned by the function, but instead be stored inside the array nums1.
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        # using two pointers from the back, gradually move numbers to the back of nums1
        i, j = m-1, n-1
        for k in reversed(range(m+n)):
            if i < 0:
                nums1[k] = nums2[j]
                j -= 1
                continue
            if j < 0:
                nums1[k] = nums1[i]
                i -= 1
                continue

            if nums1[i] > nums2[j]:
                nums1[k] = nums1[i]
                i -= 1
            else:
                nums1[k] = nums2[j]
                j -= 1

    def test(self):
        nums1 = [1, 2, 3, 0, 0, 0]
        self.merge(nums1, 3, [2, 5, 6], 3)
        print_assert(nums1, [1, 2, 2, 3, 5, 6])

        nums1 = [0]
        self.merge(nums1, 0, [1], 1)
        print_assert(nums1, [1])

        nums1 = [1]
        self.merge(nums1, 1, [], 0)
        print_assert(nums1, [1])

        nums1 = [2, 0]
        self.merge(nums1, 1, [1], 1)
        print_assert(nums1, [1, 2])


class LRUCache:

    def __init__(self, capacity: int):
        self.keys = deque([])  # right is recent, left is old
        self.map = {}
        self.capacity = capacity

    def get(self, key: int) -> int:
        if key in self.map:
            # bump
            self.keys.remove(key)
            self.keys.append(key)
            return self.map[key]
        else:
            return -1

    def put(self, key: int, value: int) -> None:
        if key in self.map:
            # perform an update. no one is evicted.
            # current key is bumped.
            self.map[key] = value # update
            self.keys.remove(key)
            self.keys.append(key)  # bump
        else:
            # add a new key. oldest is evicted.
            self.map[key] = value
            self.keys.append(key)
            if len(self.keys) > self.capacity:
                del self.map[self.keys.popleft()]


class Q146:
    def test(self):
        lru = LRUCache(2)
        lru.put(1, 1)
        lru.put(2, 2)
        print_assert(lru.get(1), 1)
        lru.put(3, 3)
        print_assert(lru.get(2), -1)
        lru.put(4, 4)
        print_assert(lru.get(1), -1)
        print_assert(lru.get(3), 3)
        print_assert(lru.get(4), 4)


if __name__ == '__main__':
    Q146().test()
