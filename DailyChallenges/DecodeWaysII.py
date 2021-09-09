from typing import List


class Solution:
    def __init__(self):
        self.cache = {}
    def numDecodings_NoStar(self, s: str) -> int:
        if len(s) <= 1:
            return 1
        if s[0] > '2' or s[0] == '2' and s[1] > '6':
            # only one way to decode the first character, move on
            return self.numDecodings_NoStar(s[1:])
        return self.numDecodings_NoStar(s[1:]) + self.numDecodings_NoStar(s[2:])

    def numDecodings(self, s: str) -> int:
        c = self.cache.get(s, -1)
        if c > -1:
            return c

        if len(s) == 0:
            ret = 1
        elif s[0] == '0':
            ret = 0
        elif s[0] == '*':
            # expand
            ret = self.numDecodings('1'+s[1:]) + self.numDecodings('2'+s[1:]) + self.numDecodings('3'+s[1:])*7
        elif len(s) == 1:
            ret = 1

        elif s[0] == '1':
            # always two ways
            if len(s) > 1 and s[1] == '*':
                ret = self.numDecodings(s[1:]) + self.numDecodings(s[2:]) * 9
            else:
                ret = self.numDecodings(s[1:]) + self.numDecodings(s[2:])
        elif s[0] == '2':
            if len(s) > 1 and s[1] == '*':
                ret = self.numDecodings(s[1:]) + self.numDecodings(s[2:]) * 6
            elif len(s) > 1 and s[1] <= '6':
                # two ways
                ret = self.numDecodings(s[1:]) + self.numDecodings(s[2:])
            else:
                ret = self.numDecodings(s[1:])
        # only one way to decode the first character, move on
        else:
            ret = self.numDecodings(s[1:])

        self.cache[s] = ret
        return ret % int(1e9+7)


if __name__ == '__main__':
    sol = Solution()
    print(sol.numDecodings('*********'), None)
    s = 0
    # for a in '123456789':
    #     for b in '123456789':
    #         for c in '123456789':
    #             for d in '123456789':
    #                 for e in '123456789':
    #                     s += sol.numDecodings(a+b+c+d+e)
    print(s)
