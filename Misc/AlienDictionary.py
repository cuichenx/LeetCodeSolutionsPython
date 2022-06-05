from typing import List, Optional
from helpers import print_assert, ListNode, LinkedList as ll


class Q269:
    # 269. Alien Dictionary
    def alienOrder(self, words: List[str]) -> str:
        NUM_LETTERS = 26  # 26 letters
        # generate graph
        adj = [[0]*NUM_LETTERS for _ in range(NUM_LETTERS)]
        letters_used = [0]*NUM_LETTERS

        def l2i(letter):
            return ord(letter)-ord('a')

        def i2l(i):
            return chr(i+ord('a'))

        def induce_order(words):
            fatal = False
            if not words or not any(words):
                return
            if words[0]:
                initials = [words[0][0]]
                letters_used[l2i(words[0][0])] = 1
            else:
                initials = ['']
            beg = 0
            for i in range(1, len(words)):
                if not words[i]:
                    fatal = True  # this means EOW token is ranked after some letters, so it is not a lexicographical order.
                    break
                letters_used[l2i(words[i][0])] = 1
                if words[i][0] != initials[-1]:
                    initials.append(words[i][0])
                    if induce_order([word[1:] for word in words[beg:i]]):
                        return True
                    beg = i
            if induce_order([word[1:] for word in words[beg:]]):
                return True
            # record order in initials
            if not fatal and len(initials) >= 2:
                for i in range(len(initials)-1):
                    if initials[i] and initials[i+1]:
                        adj[l2i(initials[i])][l2i(initials[i+1])] = 1
            return fatal

        def topological_sort():
            # find a list of start nodes
            L = []
            S = []
            for j in range(NUM_LETTERS):
                # if letter j has no incoming edges (but is a valid letter in this alphabet, i.e. has outgoing edge)
                if not any(adj[i][j] for i in range(NUM_LETTERS)) and letters_used[j]:
                    S.append(j)
            while len(S) > 0:
                cur_node = S.pop()
                L.append(cur_node)
                for j in range(NUM_LETTERS):
                    if adj[cur_node][j]:  # for every outgoing neighbour of cur_node
                        adj[cur_node][j] = 0
                        if sum(adj[i][j] for i in range(NUM_LETTERS)) == 0:  # if the neighbour has no more incoming edges
                            S.append(j)
            if sum([adj[i][j] for i in range(NUM_LETTERS) for j in range(NUM_LETTERS)]) > 0:
                return ''
            else:
                return ''.join([i2l(i) for i in L])

        if not induce_order(words):
            return topological_sort()
        else:
            return ''

    def test(self):
        print_assert(self.alienOrder(["wrt","wrf","er","ett","rftt"]), "wertf")
        print_assert(self.alienOrder(["z", "x"]), "zx")
        print_assert(self.alienOrder(["z", "x", "z"]), "")
        print_assert(self.alienOrder(["z", "z"]), "z")
        print_assert(self.alienOrder(["abc", "ab"]), "")
        print_assert(self.alienOrder(["ab", "abc"]), "cba")
        print_assert(self.alienOrder(["bc","b","cbc"]), "")

class Q19:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        # find out the length of the linked list
        if head is None:
            return None

        l = 0
        cur = head
        while cur:
            cur = cur.next
            l += 1

        if n > l:
            return head
        if n == l:
            return head.next

        ## n < l
        cur = head
        for i in range(l - n - 1):
            cur = cur.next

        cur.next = cur.next.next  # delete (l-n-1)th node from the beginning

        return head

    def test(self):
        print_assert(ll.printLinkedList(self.removeNthFromEnd(ll.makeLinkedList([1, 2, 3, 4, 5]), 2)), [1, 2, 3, 5])
        print_assert(ll.printLinkedList(self.removeNthFromEnd(ll.makeLinkedList([1]), 1)), [])
        print_assert(ll.printLinkedList(self.removeNthFromEnd(ll.makeLinkedList([1, 2]), 1)), [1])
        print_assert(ll.printLinkedList(self.removeNthFromEnd(ll.makeLinkedList([1, 2]), 1)), [1])

if __name__ == '__main__':
    Q19().test()