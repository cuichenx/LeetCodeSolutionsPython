from helpers import print_assert


class Solution:
    # 838 Push Dominoes
    # There are n dominoes in a line, and we place each domino vertically upright.
    # In the beginning, we simultaneously push some of the dominoes either to the left or to the right.

    # After each second, each domino that is falling to the left pushes the adjacent domino on the left.
    # Similarly, the dominoes falling to the right push their adjacent dominoes standing on the right.
    #
    # When a vertical domino has dominoes falling on it from both sides, it stays still due to the balance of the forces.
    #
    # For the purposes of this question, we will consider that a falling domino expends no additional
    # force to a falling or already fallen domino.
    #
    # You are given a string dominoes representing the initial state where:
    #
    # dominoes[i] = 'L', if the ith domino has been pushed to the left,
    # dominoes[i] = 'R', if the ith domino has been pushed to the right, and
    # dominoes[i] = '.', if the ith domino has not been pushed.
    # Return a string representing the final state.
    def pushDominoes(self, dominoes: str) -> str:
        # idea:
        # if the first non-dot domino is L, then all the dots in front become L, and reset after this L
        # if the first non-dot domino is R, then forget about all the dots
        #       if the second non-dot domino is R, then push all the in-between dots to R, and reset
        #       if the second non-dot domino is L, then rendezvous in the middle, and reset
        N = len(dominoes)
        res = list(dominoes)
        start, cur = 0, 0
        while True:
            while cur < N and dominoes[cur] == '.':
                cur += 1
            if cur == N:
                return ''.join(res)
            if dominoes[cur] == 'L':
                res[start:cur] = ['L'] * (cur-start)
                # reset
                start, cur = cur+1, cur+1

            else:  # dominoes[cur] == 'R'
                while cur < N and dominoes[cur] == 'R':
                    start, cur = cur, cur+1
                    while cur < N and dominoes[cur] == '.':
                        cur += 1
                    if cur >= N:
                        # set all the last few dominoes to R
                        res[start:cur] = ['R'] * (cur-start)
                        return ''.join(res)
                    if dominoes[cur] == 'L':
                        # rendezvous in the middle
                        mid = (start + cur) / 2  # could be integer or .5 decimal
                        r_end = int(mid-0.1)+1  # 5->5, 5.5->6
                        l_beg = int(mid) + 1  # 5->6, 5.5->6
                        res[start:r_end] = ['R'] * (r_end-start)
                        res[l_beg:cur] = ['L'] * (cur-l_beg)
                        # reset
                        start, cur = cur+1, cur+1
                    else:  # dominoes[cur] == 'R'
                        res[start:cur] = ['R'] * (cur-start)
                        start = cur
                        # go back to inner while loop
    # Time: O(N)  Space: O(N)

if __name__ == '__main__':
    sol = Solution()
    print_assert(sol.pushDominoes("RR.L"), "RR.L")
    print_assert(sol.pushDominoes(".L.R...LR..L.."), "LL.RR.LLRRLL..")
    print_assert(sol.pushDominoes(".RR..L.RRL.LR.LR.RL."), ".RRRLL.RRLLLR.LRRRL.")
