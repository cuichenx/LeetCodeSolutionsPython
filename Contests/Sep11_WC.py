import bisect
import math
import time
from collections import defaultdict
from typing import List

from helpers import print_assert
import heapq
import numpy as np

class Solution:
    def reversePrefix(self, word: str, ch: str) -> str:
        i = word.find(ch)
        if i == -1:
            return word
        else:
            return ''.join(list(word[:i+1])[::-1] + list(word[i+1:]))


    def test1(self):
        print_assert(self.reversePrefix('abcdefd', 'd'), 'dcbaefd')
        print_assert(self.reversePrefix('xyxzxe', 'z'), 'zxyxxe')
        print_assert(self.reversePrefix('abcd', 'z'), 'abcd')

    def interchangeableRectangles(self, rectangles: List[List[int]]) -> int:
        groups = defaultdict(int)
        for w, h in rectangles:
            groups[w/h] += 1
        num_pairs = 0
        for group_size in groups.values():
            if group_size >= 2:
                num_pairs += math.factorial(group_size)/(math.factorial(group_size-2)*2)  # group_size choose 2
        return int(num_pairs)

    def test2(self):
        print_assert(self.interchangeableRectangles([[4,8],[3,6],[10,20],[15,30]]), 6)
        print_assert(self.interchangeableRectangles([[4,5],[7,8]]), 0)

    def findPalindromes(self, s: str) -> List[str]:
        if len(s) == 1:
            return [s, '']
        if len(s) == 0:
            return ['']
        res = ['']
        for i in range(len(s)):
            for j in range(len(s)-1, i-1, -1):
                if i == j:
                    res.extend([s[i]])
                elif s[i] == s[j]:
                    res.extend([s[i]+inner+s[j] for inner in self.findPalindromes(s[i+1:j])])
        return res

    def findPalindromesIndex(self, s: str, start: int=0, end: int=None) -> List[List[int]]:
        if end is None:
            end = len(s)
        if end - start == 1:
            return [[start], []]
        if end == start:
            return [[]]
        res = [[]]
        for i in range(start, end):
            for j in range(end - 1, i-1, -1):
                if i == j:
                    res.append([i])
                elif s[i] == s[j]:
                    res.extend([[i] + inner + [j] for inner in self.findPalindromesIndex(s, i+1, j)])
        return res

    def maxProduct(self, s: str) -> int:
        # 3 seconds
        palin_indices = self.findPalindromesIndex(s)
        palin_indices.sort(key=lambda l: len(l), reverse=True)
        cur_max_prod = 0
        tot_len = len(palin_indices)
        j = tot_len - 1
        for i in range(tot_len-1):
            cand1 = set(palin_indices[i])
            if len(cand1)**2 <= cur_max_prod:
                break  # no need to check anymore
            for j in range(i+1, tot_len):
                cand2 = set(palin_indices[j])
                if len(cand1) + len(cand2) > tot_len:
                    continue
                if len(cand1.union(cand2)) == len(cand1) + len(cand2): # no overlap
                    cur_max_prod = max(cur_max_prod, len(cand1)*len(cand2))
                j -= 1
        return cur_max_prod

    def maxProduct_v2(self, s: str) -> int:
        # 6 seconds
        palin_indices = self.findPalindromesIndex(s)
        palin_indices_d = defaultdict(list)
        for p in palin_indices:
            palin_indices_d[len(p)].append(set(p))
        tot_len = len(palin_indices)
        order = []
        for i in range(1, tot_len-2):
            for j in range(i, tot_len):
                if i+j <= tot_len:
                    order.append([i, j])
        order.sort(key=lambda x: x[0]*x[1], reverse=True)

        for (len1, len2) in order:
            for cand1 in palin_indices_d[len1]:
                for cand2 in palin_indices_d[len2]:
                    if len(cand1.union(cand2)) == len1 + len2:
                        return len1*len2
        return 1

    def maxProduct_v3(self, s: str) -> int:
        # 1 second
        palin_indices = self.findPalindromesIndex(s)
        palins_d = {}
        for palin in palin_indices:
            bits = 0
            for p in palin:
                bits += 1<<p  # bits += 2**p
            palins_d[bits] = len(palin)

        cur_max_prod = 0
        for cand1, len1  in palins_d.items():
            for cand2, len2 in palins_d.items():
                if cand1 & cand2 == 0:
                    cur_max_prod = max(cur_max_prod, len1*len2)
        return cur_max_prod

    def test3(self):
        print_assert(self.maxProduct('leetcodecom'), 9)
        print_assert(self.maxProduct('bb'), 1)
        print_assert(self.maxProduct('abcdefghijkl'), 1)
        print_assert(self.maxProduct('ueeu'), 4)
        print_assert(self.maxProduct('abccba'), 9)
        print_assert(self.maxProduct('zlpzl'), 6)
        t0 = time.time()
        print_assert(self.maxProduct_v3('e'*12), 36)
        print(time.time()-t0, 'seconds')


if __name__ == '__main__':
    sol = Solution()
    sol.test3()


