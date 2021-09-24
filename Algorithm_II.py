import bisect
import math
import time
from collections import defaultdict, Counter, deque
from typing import List, Optional

from helpers import print_assert, ListNode, LinkedList as ll, Tree as tr
import heapq


class AlgoIIDay1:
    # 34. Find First and Last Position of Element in Sorted Array
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        idx1 = bisect.bisect_left(nums, target)
        if idx1 < len(nums) and nums[idx1] == target:
            idx2 = bisect.bisect_right(nums, target)
            return [idx1, idx2 - 1]
        else:
            return [-1, -1]

    def test1(self):
        print_assert(self.searchRange([5, 7, 7, 8, 8, 10], 8), [3, 4])
        print_assert(self.searchRange([5, 7, 7, 8, 8, 10], 6), [-1, -1])
        print_assert(self.searchRange([], 0), [-1, -1])

    # 33. Search in Rotated Sorted Array
    def search(self, nums: List[int], target: int) -> int:
        # two binary searches
        # first one to locate the minimum element (orig start)
        # second one to do normal binary search, given adjusted indices
        n = len(nums)
        lo, hi = -1, n - 1
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if nums[mid] <= nums[hi]:
                # second half is sorted.
                hi = mid
            else:
                # first half is sorted
                lo = mid
        orig_start = hi

        # print(orig_start)
        def f(x):
            # orig index in sorted array to new index in rotated array
            return (x + orig_start) % n

        lo, hi = 0, n  # lo mid hi are rotated indices
        while lo < hi:
            mid = (lo + hi) // 2
            if target <= nums[f(mid)]:
                hi = mid
            else:
                lo = mid + 1
        if nums[f(lo)] == target:
            return f(lo)
        else:
            return -1

    def test2(self):
        print_assert(self.search([0], 0), 0)
        print_assert(self.search([0, 1], 0), 0)
        print_assert(self.search([0, 1, 2], 0), 0)
        print_assert(self.search([0, 1, 2, 3], 0), 0)
        print_assert(self.search([0, 1, 2, 3, 4], 0), 0)
        print_assert(self.search([4, 5, 6, 7, 0, 1, 2], 3), -1)
        print_assert(self.search([3, 1], 1), 1)
        print_assert(self.search([1, 3], 1), 0)
        print_assert(self.search([5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4], 7), 2)
        print_assert(self.search([5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4], 3), 10)
        print_assert(self.search([5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4], 7), 2)
        print_assert(self.search([5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4], 3), 9)
        print_assert(self.search([5, 6, 7, 8, 9, 0, 1, 2, 3, 4], 7), 2)
        print_assert(self.search([5, 6, 7, 8, 9, 0, 1, 2, 3, 4], 3), 8)
        print_assert(self.search([5, 6, 7, 8, 0, 1, 2, 3, 4], 7), 2)
        print_assert(self.search([5, 6, 7, 8, 0, 1, 2, 3, 4], 3), 7)
        print_assert(self.search([5, 6, 7, 0, 1, 2, 3, 4], 7), 2)
        print_assert(self.search([5, 6, 7, 0, 1, 2, 3, 4], 3), 6)
        print_assert(self.search([5, 6, 0, 1, 2, 3, 4], 7), -1)
        print_assert(self.search([5, 6, 0, 1, 2, 3, 4], 3), 5)
        print_assert(self.search([5, 0, 1, 2, 3, 4], 7), -1)
        print_assert(self.search([5, 0, 1, 2, 3, 4], 3), 4)
        print_assert(self.search([0, 1, 2, 3, 4], 7), -1)
        print_assert(self.search([0, 1, 2, 3, 4], 3), 3)

    # 74. Search a 2D Matrix
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        m, n = len(matrix), len(matrix[0])

        lo, hi = 0, m*n  # lo mid hi are rotated indices
        while lo < hi:
            mid = (lo + hi) // 2
            if target <= matrix[mid//n][mid%n]:
                hi = mid
            else:
                lo = mid + 1
        return lo < m*n and matrix[lo//n][lo%n] == target

    def test3(self):
        print_assert(self.searchMatrix([[1,3,5,7],[10,11,16,20],[23,30,34,60]], 3), True)
        print_assert(self.searchMatrix([[1,3,5,7],[10,11,16,20],[23,30,34,60]], 13), False)
        print_assert(self.searchMatrix([[1]], 1), True)
        print_assert(self.searchMatrix([[1]], 2), False)


class AlgoIIDay2:
    def findMin(self, nums: List[int]) -> int:
        # similar to yesterday's question 2??
        n = len(nums)
        lo, hi = -1, n - 1
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if nums[mid] <= nums[hi]:
                # second half is sorted.
                hi = mid
            else:
                # first half is sorted
                lo = mid
        return nums[hi]

    def test1(self):
        print_assert(self.findMin([3,4,5,1,2]), 1)
        print_assert(self.findMin([4,5,6,7,0,1,2]), 0)
        print_assert(self.findMin([11,13,15,17]), 11)
        print_assert(self.findMin([1,11,13,15,17]), 1)

    def findPeakElement(self, nums: List[int]) -> int:
        n = len(nums)
        # if lo > mid, then first half must contain a peak, or nums[0] is a peak
        # elif mid < hi, then second half must contain a peak, or nums[n-1] is a peak
        # else, (lo < mid and mid > hi), check go to the higher one of mid-1 and mid+1
        lo, hi = 0, n-1
        while lo < hi - 1:
            mid = (lo+hi)//2
            if nums[mid] > nums[mid-1] and nums[mid] > nums[mid+1]:
                return mid
            if nums[lo] > nums[mid]:
                hi = mid
            elif nums[hi] > nums[mid]:
                lo = mid
            elif nums[mid-1] > nums[mid]:
                hi = mid
            else:
                lo = mid
        return lo if nums[lo]>nums[hi] else hi

    def test2(self):
        print_assert(self.findPeakElement([1, 2, 3, 1]), 2)
        print_assert(self.findPeakElement([1, 2, 1, 3, 5, 6, 4]), (5, 1))
        print_assert(self.findPeakElement([1, 2, 3, 4, 5, 6, 7, 8, 7]), 7)
        print_assert(self.findPeakElement([1, 2, 3, 4, 5, 6, 7, 8, 9]), 8)
        print_assert(self.findPeakElement([3, 2, 1]), 0)
        print_assert(self.findPeakElement([3, 4, 2, 1]), 1)


class AlgoIIDay3:
    def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # fist handle the case where first few elements are duplicates
        while True:
            first_elements_duplicates = False
            while head and head.next and head.val == head.next.val:
                head = head.next
                first_elements_duplicates = True
            if first_elements_duplicates:
                head = head.next  # and continue. the next head might also be duplicates
            else:
                break
        # now first few elements aren't duplicates

        keep = None
        cur = head
        while cur:
            if keep is None:
                # looking for duplicate nodes
                if not cur.next or not cur.next.next or cur.next.val != cur.next.next.val:  # no duplicate
                    cur = cur.next
                else:
                    # duplicate incoming, keep track of cur, to be linked to a later node.
                    keep = cur
            else:
                # looking at duplicate nodes here
                if (cur.next and cur.val == cur.next.val) or \
                        (cur.next and cur.next.next and cur.next.val == cur.next.next.val):
                    cur = cur.next   # still in a run of duplicates, keep advancing
                else:
                    # run of duplicates ended
                    keep.next = cur.next
                    keep = None
                    cur = cur.next
        if keep is not None:
            # if list finished while looking for more duplicates, then modified list ends with keep.
            keep.next = None
        return head

    def test1(self):
        print_assert(ll.printLinkedList(self.deleteDuplicates(ll.makeLinkedList([1, 2, 3, 3, 4, 4, 5]))), [1, 2, 5])
        print_assert(ll.printLinkedList(self.deleteDuplicates(ll.makeLinkedList([1, 1, 1, 2, 3]))), [2, 3])
        print_assert(ll.printLinkedList(self.deleteDuplicates(ll.makeLinkedList([1, 3, 3, 4, 4, 4, 4, 4, 5, 7, 7, 8, 9]))), [1, 5, 8, 9])
        print_assert(ll.printLinkedList(self.deleteDuplicates(ll.makeLinkedList([1, 2, 3, 4]))), [1, 2, 3, 4])
        print_assert(ll.printLinkedList(self.deleteDuplicates(ll.makeLinkedList([1, 1, 2, 3, 4]))), [2, 3, 4])
        print_assert(ll.printLinkedList(self.deleteDuplicates(ll.makeLinkedList([]))), [])
        print_assert(ll.printLinkedList(self.deleteDuplicates(ll.makeLinkedList([1, 1, 2, 2]))), [])

    def threeSum(self, nums: List[int]) -> List[List[int]]:
        ret = []
        nums.sort()
        n = len(nums)
        for i in range(n):
            if i > 0 and nums[i] == nums[i-1]:
                continue  # don't calculate duplicate i
            k = n - 1
            j = i + 1
            while j < k:
                cur_sum = nums[i] + nums[j] + nums[k]
                if cur_sum > 0:
                    k -= 1
                elif cur_sum < 0:
                    j += 1
                else:
                    ret.append([nums[i], nums[j], nums[k]])
                    # increment j until it's different. decrement k until it's different
                    j += 1
                    while j < k and nums[j-1] == nums[j]: j += 1
                    k -= 1
                    while k > i and nums[k+1] == nums[k]: k -= 1
        return ret

    def test2(self):
        print_assert(self.threeSum([-1, 0, 1, 2, -1, -4]), [[-1, -1, 2], [-1, 0, 1]])
        print_assert(self.threeSum([]), [])
        print_assert(self.threeSum([0]), [])
        print_assert(self.threeSum([0, 0, 0]), [[0, 0, 0]])
        print_assert(self.threeSum([0, 0, 0, 0]), [[0, 0, 0]])
        print_assert(self.threeSum([0, 0, 0, 0, -1]), [[0, 0, 0]])
        print_assert(self.threeSum([0, 0, 0, 0, -1, 1]), [[-1, 0, 1], [0, 0, 0]])
        print_assert(self.threeSum([-2, 0, 1, 1, 2]), [[-2, 0, 2], [-2, 1, 1]])


class AlgoIIDay4:
    def backspaceCompare(self, s: str, t: str) -> bool:
        # O(1) space solution
        i, j = len(s) - 1, len(t) - 1
        s_backs, t_backs = 0, 0
        s_let, t_let = '', ''
        while i >= 0 or j >= 0:
            if i >= 0 and not s_let:  # don't go in here if s_let is not cleared
                if s[i] == '#':
                    s_backs += 1
                elif s_backs > 0:  # if s[i] is a letter and there are pending backspaces
                    s_backs -= 1
                else:
                    s_let = s[i]
                i -= 1
            if i < 0 and not s_let:
                s_let = '^'  # begin symbol, so if there are extra letters in t the comparison would return false

            if j >= 0 and not t_let:
                if t[j] == '#':
                    t_backs += 1
                elif t_backs > 0:
                    t_backs -= 1
                else:
                    t_let = t[j]
                j -= 1
            if j < 0 and not t_let:
                t_let = '^'  # begin symbol, so if there are extra letters in s the comparison would return false

            if s_let and t_let:
                if s_let == t_let:
                    s_let, t_let = '', ''  # same letter. can continue
                else:
                    return False

        return True

    def test1(self):
        print_assert(self.backspaceCompare("ab#c", "ad#c"), True)
        print_assert(self.backspaceCompare("ab##", "c#d#"), True)
        print_assert(self.backspaceCompare("a##c", "#a#c"), True)
        print_assert(self.backspaceCompare("a#c", "b"), False)
        print_assert(self.backspaceCompare("a#c", "c"), True)
        print_assert(self.backspaceCompare("c", "c"), True)
        print_assert(self.backspaceCompare("bc", "c"), False)
        print_assert(self.backspaceCompare("b", "c"), False)
        print_assert(self.backspaceCompare("c#", "c"), False)
        print_assert(self.backspaceCompare("c###", "c#"), True)
        print_assert(self.backspaceCompare("abc#", "#####ab"), True)

    def intervalIntersection(self, firstList: List[List[int]], secondList: List[List[int]]) -> List[List[int]]:
        def get_overlap(first_interval: List[int], second_interval: List[int]) -> Optional[List[int]]:
            if first_interval[1] < second_interval[0] or second_interval[1] < first_interval[0]:
                return None  # no overlap
            elif first_interval[1] > second_interval[1] and second_interval[0] > first_interval[0]:
                # first interval shadows second
                return second_interval
            elif second_interval[1] > first_interval[1] and first_interval[0] > second_interval[0]:
                # second interval shadows first
                return first_interval
            else:
                return [max(first_interval[0], second_interval[0]), min(first_interval[1], second_interval[1])]

        i, j = 0, 0
        ret = []
        while i < len(firstList) and j < len(secondList):
            overlap = get_overlap(firstList[i], secondList[j])
            if overlap:
                ret.append(overlap)
            if firstList[i][1] > secondList[j][1]:
                j += 1
            else:
                i += 1
        return ret

    def test2(self):
        print_assert(self.intervalIntersection([[0,2],[5,10],[13,23],[24,25]], [[1,5],[8,12],[15,24],[25,26]]),
                     [[1,2],[5,5],[8,10],[15,23],[24,24],[25,25]])
        print_assert(self.intervalIntersection([[1,3],[5,9]], []), [])
        print_assert(self.intervalIntersection([], [[1,3],[5,9]]), [])
        print_assert(self.intervalIntersection([[1,7]], [[3,10]]), [[3,7]])
        print_assert(self.intervalIntersection([[1,7]], [[3,10]]), [[3,7]])
        print_assert(self.intervalIntersection([[5, 10]], [[3,10]]), [[5,10]])
        print_assert(self.intervalIntersection([[3, 10]], [[5,10]]), [[5,10]])
        print_assert(self.intervalIntersection([[3,5],[9,20]], [[4,5],[7,10],[11,12],[14,15],[16,20]]),
                     [[4,5],[9,10],[11,12],[14,15],[16,20]])

    def maxArea_wrong(self, height: List[int]) -> int:
        n = len(height)
        left = [(height[0], 0)]
        right = [(height[-1], 0)]  # order is reversed
        for i in range(1, n):
            if height[i] > left[-1][0]:
                left.append((height[i], 0))
            else:
                left.append((left[-1][0], left[-1][1]+1))
            if height[-i-1] > right[-1][0]:
                right.append((height[-i-1], 0))
            else:
                right.append((right[-1][0], right[-1][1]+1))
        cur_max = 0
        for (lh, li), (rh, ri) in zip(left, reversed(right)):
            cur_max = max(cur_max, min(lh, rh) * (li + ri))
        return cur_max

    def maxArea(self, height: List[int]) -> int:
        n = len(height)
        i, j = 0, n-1
        cur_max = 0
        while j > i:
            cur_max = max(cur_max, (j - i) * min(height[i], height[j]))
            if height[i] < height[j]:
                # try to get a higher left beam
                i += 1
            else:
                j -= 1
        return cur_max

    def test3(self):
        print_assert(self.maxArea([1,8,6,2,5,4,8,3,7]), 49)
        print_assert(self.maxArea([1, 1]), 1)
        print_assert(self.maxArea([4, 3, 2, 1, 4]), 16)
        print_assert(self.maxArea([1, 2, 1]), 2)


class AlgoIIDay5:  # sliding window
    def findAnagrams(self, s: str, p: str) -> List[int]:
        p_bag = Counter(p)
        window = Counter(s[:len(p)])
        ret = []
        for i in range(len(s) - len(p)):
            if window == p_bag:
                ret.append(i)

            window[s[i+len(p)]] += 1
            window[s[i]] -= 1
            if window[s[i]] == 0:
                del window[s[i]]

        if window == p_bag:
            ret.append(len(s)-len(p))
        return ret

    def test1(self):
        print_assert(self.findAnagrams("cbaebabacd", "abc"), [0, 6])
        print_assert(self.findAnagrams("abab", "ab"), [0, 1, 2])
        print_assert(self.findAnagrams("ab", "abc"), [])
        print_assert(self.findAnagrams("abc", "abcafasfasf"), [])

    def numSubarrayProductLessThanK_slow(self, nums: List[int], k: int) -> int:
        # n^2, slow!!
        cumul_log_prod = [0]
        if k == 0:
            return 0
        logk = math.log10(k)
        count = 0
        for num in nums:
            cumul_log_prod.append(cumul_log_prod[-1]+math.log10(num))  # num will never be negative
        for i in range(len(nums)):
            for j in range(i+1, len(nums)+1):
                if cumul_log_prod[j] - cumul_log_prod[i] < logk - 1e-9:
                    count += 1
        return count

    def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
        if k == 0:
            return 0
        i = 0
        count = 0
        window_prod = 1
        # following hint, for each j, find the largest window (left most i) whose product is less than k
        for j in range(len(nums)):
            window_prod *= nums[j]  # add nums[j] to window
            while window_prod >= k and i <= j:
                window_prod /= nums[i]  # remove nums[i] from window
                i += 1
            # now window_prod < k, or i = j+1
            count += (j+1-i)
        return count

    def test2(self):
        print_assert(self.numSubarrayProductLessThanK([10, 5, 2, 6], 100), 8)
        print_assert(self.numSubarrayProductLessThanK([1, 2, 3], 0), 0)
        print_assert(self.numSubarrayProductLessThanK([1, 2, 3], 1), 0)
        print_assert(self.numSubarrayProductLessThanK([1, 2, 3], 2), 1)
        print_assert(self.numSubarrayProductLessThanK([1, 2, 3], 3), 3)
        print_assert(self.numSubarrayProductLessThanK([1, 2, 3], 4), 4)
        print_assert(self.numSubarrayProductLessThanK([1, 2, 3], 5), 4)
        print_assert(self.numSubarrayProductLessThanK([1, 2, 3], 6), 4)
        print_assert(self.numSubarrayProductLessThanK([1, 2, 3], 7), 6)

    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        i, j = 0, 1  # range [i, j)
        min_len = len(nums)+1
        window_sum = nums[0]
        while j <= len(nums):
            if window_sum >= target:
                min_len = min(min_len, j-i)
                window_sum -= nums[i]
                i += 1  # try a shorted window
            elif j == len(nums):
                break
            else:
                window_sum += nums[j]
                j += 1  # try a longer window
        if min_len == len(nums)+1:
            return 0
        return min_len

    def test3(self):
        print_assert(self.minSubArrayLen(7, [2, 3, 1, 2, 4, 3]), 2)
        print_assert(self.minSubArrayLen(7, [2, 3, 1, 2, 4, 3, 1]), 2)
        print_assert(self.minSubArrayLen(7, [2, 3, 1, 2, 4, 3, 7]), 1)
        print_assert(self.minSubArrayLen(4, [1, 4, 4]), 1)
        print_assert(self.minSubArrayLen(11, [1, 1, 1, 1, 1]), 0)


class AlgoIIDay6:
    def numIslands(self, grid: List[List[str]]) -> int:
        q = deque([])
        m, n = len(grid), len(grid[0])
        island_count = 0
        for global_i in range(m):
            for global_j in range(n):
                if grid[global_i][global_j] == '1':
                    # new island found
                    island_count += 1
                    q.append((global_i, global_j))
                    while len(q) > 0:
                        i, j = q.popleft()
                        if grid[i][j] == '1':
                            grid[i][j] = 'x'
                            if i > 0 and grid[i-1][j] == '1':
                                q.append((i-1, j))  # up; this probably won't happen
                            if i < m-1 and grid[i+1][j] == '1':
                                q.append((i+1, j))  # down
                            if j > 0 and grid [i][j-1] == '1':
                                q.append((i, j-1))  # left
                            if j < n-1 and grid[i][j+1] == '1':
                                q.append((i, j+1))
        return island_count
    # Time: O(MN)
    # Space: O(MN)
    def test1(self):
        print_assert(self.numIslands([
                      ["1","1","1","1","0"],
                      ["1","1","0","1","0"],
                      ["1","1","0","0","0"],
                      ["0","0","0","0","0"]
                    ]), 1)
        print_assert(self.numIslands([
                      ["1","1","0","0","0"],
                      ["1","1","0","0","0"],
                      ["0","0","1","0","0"],
                      ["0","0","0","1","1"]
                    ]), 3)

    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        explored = set([])
        q = deque([])
        n = len(isConnected)
        num_provinces = 0
        for i in range(n):
            if i in explored: continue
            # i is a new group of nodes
            num_provinces += 1
            # add neighbours of i to explored
            q.append(i)
            while len(q) > 0:
                cur_node = q.popleft()
                # if cur_node in explored: continue
                explored.add(cur_node)
                for j, is_connected in enumerate(isConnected[cur_node]):
                    if is_connected and j not in explored:
                        q.append(j)
        return num_provinces

    def test2(self):
        print_assert(self.findCircleNum([[1,1,0],[1,1,0],[0,0,1]]), 2)
        print_assert(self.findCircleNum([[1,0,0],[0,1,0],[0,0,1]]), 3)
        print_assert(self.findCircleNum([[1,1,0],[1,1,1],[0,1,1]]), 1)


class AlgoIIDay7:
    from helpers import TreeNode
    def connect(self, root: 'TreeNode') -> 'TreeNode':
        # BFS on tree
        if root is None:
            return root
        cur_level = []
        next_level = [root]
        while len(next_level) > 0:
            cur_level, next_level = next_level, []
            # do the next connections
            for i in range(len(cur_level)-1):
                cur_level[i].next = cur_level[i+1]
            cur_level[-1].next = None

            # get the next level ready
            for node in cur_level:
                if node.left:
                    next_level.append(node.left)
                if node.right:
                    next_level.append(node.right)
        return root
# Time O(N)
# Space O(N)
# Space optimization: instead of saving the "next_level", save only the first element of next level
# use the next pointer to fill up the row

    def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
        # do bfs of tree, do treeEqual on each node
        q = deque([root])
        while len(q) > 0:
            cur_node = q.popleft()
            if self.treeEqual(cur_node, subRoot):
                return True
            if cur_node:
                if cur_node.left:
                    q.append(cur_node.left)
                if cur_node.right:
                    q.append(cur_node.right)
        return False


    def treeEqual(self, root: Optional[TreeNode], other: Optional[TreeNode]) -> bool:
        if root is None and other is None:
            return True
        elif root is None or other is None:
            return False
        elif root.val == other.val:
            return self.treeEqual(root.left, other.left) and self.treeEqual(root.right, other.right)
        else:
            return False

    def test2(self):
        null = None
        print_assert(self.isSubtree(tr.list2tree([3, 4, 5, 1, 2]), tr.list2tree([4, 1, 2])), True)
        print_assert(self.isSubtree(tr.list2tree([3,4,5,1,2,null,null,null,null,0]), tr.list2tree([4, 1, 2])), False)
        print_assert(self.isSubtree(tr.list2tree([3]), tr.list2tree([4])), False)
        print_assert(self.isSubtree(tr.list2tree([4]), tr.list2tree([4])), True)

if __name__ == '__main__':
    AlgoIIDay7().test2()
