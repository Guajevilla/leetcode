# ############################### 241. 为运算表达式设计优先级 #################################
x = "2-1-1"         # [0, 2]
x = "2*3-4*5"       # [-34, -14, -10, -10, 10]


class Solution:
    # def diffWaysToCompute(self, input: str):
    #     if not input:
    #         return []
    #     elif input.isdigit():
    #         return [int(input)]
    #     res = []
    #     for i, ele in enumerate(input):
    #         # 这里尝试用了一下列表表达式,也可写开双重循环append
    #         if ele == '+':
    #             res.extend([a + b for a in self.diffWaysToCompute(input[:i]) for b in self.diffWaysToCompute(input[i+1:])])
    #         elif ele == '-':
    #             res.extend([a - b for a in self.diffWaysToCompute(input[:i]) for b in self.diffWaysToCompute(input[i+1:])])
    #         elif ele == '*':
    #             res.extend([a * b for a in self.diffWaysToCompute(input[:i]) for b in self.diffWaysToCompute(input[i+1:])])
    #
    #     return res

    def diffWaysToCompute(self, input):
        if input.isdigit():
            return [int(input)]
        res = []
        for i, opt in enumerate(input):
            if opt in {"+", "-", "*"}:
                left = self.diffWaysToCompute(input[:i])
                right = self.diffWaysToCompute(input[i + 1:])
                res.extend([self.helper(l, r, opt) for l in left for r in right])
        return res

    def helper(self, m, n, op):
        if op == "+":
            return m + n
        elif op == "-":
            return m - n
        else:
            return m * n


solve = Solution()
print(solve.diffWaysToCompute(x))

# ############################### 242. 有效的字母异位词 #################################
# s = "anagram"
# t = "nagaram"       #  T
#
# # s = "rat"
# # t = "car"           # F
#
#
# class Solution:
#     def isAnagram(self, s: str, t: str) -> bool:
#         if len(s) != len(t):
#             return False
#         rem_s = {}
#         rem_t = {}
#         for i in range(len(s)):
#             if s[i] in rem_s:
#                 rem_s[s[i]] += 1
#             else:
#                 rem_s[s[i]] = 1
#
#             if t[i] in rem_t:
#                 rem_t[t[i]] += 1
#             else:
#                 rem_t[t[i]] = 1
#
#         return rem_s == rem_t
#
#         # return sorted(s) == sorted(t)
#
#
# solve = Solution()
# print(solve.isAnagram(s, t))
