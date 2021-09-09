class Solution:
    def isIsomorphic_dict(self, s: str, t: str) -> bool:
        s2t = {}
        t2s = {}
        for s_letter, t_letter in zip(s, t):
            if s2t.get(s_letter, t_letter) != t_letter:
                return False
            s2t[s_letter] = t_letter
            if t2s.get(t_letter, s_letter) != s_letter:
                return False
            t2s[t_letter] = s_letter
        return True

    def isIsomorphic(self, s: str, t: str) -> bool:
        s2t = [-1] * 256
        t2s = [-1] * 256
        for s_letter, t_letter in zip(s, t):
            if s2t[ord(s_letter)] not in [-1, t_letter]:
                return False
            s2t[ord(s_letter)] = t_letter
            if t2s[ord(t_letter)] not in [-1, s_letter]:
                return False
            t2s[ord(t_letter)] = s_letter
        return True


if __name__ == '__main__':
    sol = Solution()
    print(sol.isIsomorphic('abc', 'fgh'), True)
    print(sol.isIsomorphic('abcc', 'fghh'), True)
    print(sol.isIsomorphic('aabc', 'ffgh'), True)
    print(sol.isIsomorphic('aabc', 'fghf'), False)
    print(sol.isIsomorphic('abc', 'ffa'), False)
    print(sol.isIsomorphic('143a1', 'fghgf'), False)