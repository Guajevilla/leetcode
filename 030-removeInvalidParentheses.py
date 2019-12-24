# ############################### 301. 删除无效的括号 #################################
# 删除最小数量的无效括号，使得输入的字符串有效，返回所有可能的结果。
# 说明: 输入可能包含了除 ( 和 ) 以外的字符。
s = "()())()"           # ["()()()", "(())()"]
s = "(a)())()"          # ["(a)()()", "(a())()"]
s = ")("                # [""]


class Solution:
    def removeInvalidParentheses(self, s):
        l_cnt = 0
        r_cnt = 0
        return


solve = Solution()
print(solve.removeInvalidParentheses(s))
print('##############################################################')

# ############################### 303. 区域和检索 - 数组不可变 ################################
#
# class NumArray:
#
#     def __init__(self, nums):
#         self.rem = [0] * (len(nums) + 1)
#         for i, num in enumerate(nums):
#             self.rem[i + 1] = num + self.rem[i]
#
#     def sumRange(self, i: int, j: int) -> int:
#         return self.rem[j + 1] - self.rem[i]
#
#
# # Your NumArray object will be instantiated and called as such:
# obj = NumArray([-2, 0, 3, -5, 2, -1])
# print(obj.sumRange(0,2))        # 1
# print(obj.sumRange(2,5))        # -1
# print(obj.sumRange(0,5))        # -3

# ############################# 304. 二维区域和检索 - 矩阵不可变 #############################
# class NumMatrix:
#
#     def __init__(self, matrix):
#         if not matrix or not matrix[0]:
#             self.rem = None
#         else:
#             self.rem = [[0] * (len(matrix[0]) + 1) for _ in range(len(matrix)+1)]
#             for i, lis in enumerate(matrix):
#                 for j, num in enumerate(lis):
#                     self.rem[i + 1][j + 1] = num + self.rem[i][j + 1]
#             for i, lis in enumerate(matrix):
#                 for j, num in enumerate(lis):
#                     self.rem[i + 1][j + 1] += self.rem[i + 1][j]
#         # print(self.rem)
#
#         # 也可以一遍循环求出来..
#         # for i in range(1, len(matrix) + 1):
#         #     for j in range(1, len(matrix[0]) + 1):
#         #         self.dp[i][j]=self.dp[i-1][j]+self.dp[i][j-1]+matrix[i-1][j-1]-self.dp[i-1][j-1]
#
#     def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
#         if not self.rem:
#             return 0
#         res = self.rem[row2 + 1][col2 + 1] + self.rem[row1][col1]
#         res -= self.rem[row1][col2 + 1] + self.rem[row2 + 1][col1]
#         return res
#
# # Your NumMatrix object will be instantiated and called as such:
# matrix = [
#   [3, 0, 1, 4, 2],
#   [5, 6, 3, 2, 1],
#   [1, 2, 0, 1, 5],
#   [4, 1, 0, 1, 7],
#   [1, 0, 3, 0, 5]
# ]
#
# obj = NumMatrix(matrix)
# print(obj.sumRegion(2,1,4,3))           # 8
# print(obj.sumRegion(1,1,2,2))           # 11
# print(obj.sumRegion(1,2,2,4))           # 12

# ############################### 306. 累加数 #################################
# x = "0112358"       # T
# # x = "112358"        # T
# # x = "199100199"     # T
# # x = "1091019"       # F
# x = "101"       # T
# # x = "0235813"       # F
# # x = "9817"       # T
# # x = "198019823962"      # T
# x = "10111"       # T
#
#
# class Solution:
#     def isAdditiveNumber(self, num: str) -> bool:
#         if len(num) < 3:
#             return False
#
#         def backtrack(num1, num2, s):
#             tmp_s = str(num1+num2)
#             if s[:len(tmp_s)] == tmp_s:
#                 if s[len(tmp_s):]:
#                     if backtrack(num2, num1+num2, s[len(tmp_s):]):
#                         return True
#                 else:
#                     return True
#             else:
#                 return False
#
#         # 第一个数和第二个数之间有长度要求
#         for i in range(len(num) // 2):
#             if num[0] == '0' and i > 0:
#                 return False
#             for j in range(i + 1, 2 * len(num) // 3):
#                 if num[i+1] == '0' and j - i > 1:
#                     break
#                 if backtrack(int(num[:i+1]), int(num[i+1:j+1]), num[j+1:]):
#                     return True
#         return False
#
#
# solve = Solution()
# print(solve.isAdditiveNumber(x))

# ############################### 307. 区域和检索 - 数组可修改 ##############################
# update(i, val) 函数可以通过将下标为 i 的数值更新为 val，从而对数列进行修改。
class NumArray:
    # # 超时
    # def __init__(self, nums):
    #     self.nums = nums
    #     self.rem = [0] * (len(nums) + 1)
    #     for i, num in enumerate(nums):
    #         self.rem[i + 1] = num + self.rem[i]
    #
    # def update(self, i: int, val: int) -> None:
    #     delta = val - self.nums[i]
    #     self.nums[i] = val
    #     for j in range(i + 1, len(self.rem)):
    #         self.rem[j] += delta
    #
    # def sumRange(self, i: int, j: int) -> int:
    #     return self.rem[j + 1] - self.rem[i]

    # # 线段树的数组实现
    # def __init__(self, nums):
    #     self.l = len(nums)
    #     self.tree = [0] * self.l + nums
    #     for i in range(self.l - 1, 0, -1):
    #         self.tree[i] = self.tree[i << 1] + self.tree[i << 1 | 1]
    #
    # def update(self, i, val):
    #     n = self.l + i
    #     self.tree[n] = val
    #     while n > 1:
    #         self.tree[n >> 1] = self.tree[n] + self.tree[n ^ 1]
    #         n >>= 1
    #
    # def sumRange(self, i, j):
    #     m = self.l + i
    #     n = self.l + j
    #     res = 0
    #     while m <= n:
    #         if m & 1:
    #             res += self.tree[m]
    #             m += 1
    #         m >>= 1
    #         if n & 1 == 0:
    #             res += self.tree[n]
    #             n -= 1
    #         n >>= 1
    #     return res

    # 线段树
    def __init__(self, nums):
        self.n = len(nums)
        self.s_tree = [0] * (self.n + 1)
        for i in range(1, self.n + 1):
            self.s_update(i, nums[i - 1])
        self.nums = nums

    def update(self, i: int, val: int) -> None:
        self.s_update(i + 1, val - self.nums[i])
        self.nums[i] = val

    def sumRange(self, i: int, j: int) -> int:
        return self.s_sum(j + 1) - self.s_sum(i)

    def s_update(self, i, val):
        while i <= self.n:
            self.s_tree[i] += val
            i += i & -i

    def s_sum(self, i):
        res = 0
        while i > 0:
            res += self.s_tree[i]
            i -= i & -i
        return res


# Your NumArray object will be instantiated and called as such:
# obj = NumArray([1,3,5])
# print(obj.sumRange(0, 2))       # 9
# obj.update(1, 2)
# print(obj.sumRange(0, 2))       # 8

obj = NumArray([7,2,7,2,0])
obj.update(4, 6)
obj.update(0, 2)
obj.update(0, 9)
print(obj.sumRange(4, 4))       # 6
obj.update(3, 8)
print(obj.sumRange(0, 4))       # 32
obj.update(4, 1)
print(obj.sumRange(0, 3))       # 26
print(obj.sumRange(0, 4))       # 27
obj.update(0, 4)

# ############################### 309. 最佳买卖股票时机含冷冻期 ##############################
x = [1,2,3,0,2]     # 3


class Solution:
    def maxProfit(self, prices) -> int:
        # dp[i][0]表示在i天买入最大利益
        # dp[i][1]表示在i天卖出最大利益
        # dp[i][2]表示在经过卖出的后一天冷冻期的最大利益
        if not prices: return 0
        n = len(prices)
        dp = [[0] * 3 for _ in range(n)]

        dp[0][0] = -prices[0]
        for i in range(1, n):
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][2] - prices[i])
            dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] + prices[i])
            dp[i][2] = dp[i - 1][1]
        print(dp)
        return dp[-1][1]


solve = Solution()
print(solve.maxProfit(x))

# ############################### 310. 最小高度树 ##############################
# n = 4
# edges = [[1, 0], [1, 2], [1, 3]]        # [1]
#
# n = 6
# edges = [[0, 3], [1, 3], [2, 3], [4, 3], [5, 4]]    # [3, 4]
#
# # n = 63
# # edges = [[0,1],[1,2],[0,3],[2,4],[0,5],[4,6],[1,7],[2,8],[3,9],[4,10],[1,11],[3,12],[2,13],[1,14],[8,15],[9,16],[6,17],[8,18],[4,19],[13,20],[19,21],[8,22],[19,23],[23,24],[14,25],[10,26],[3,27],[21,28],[22,29],[6,30],[26,31],[16,32],[15,33],[17,34],[3,35],[14,36],[29,37],[26,38],[34,39],[39,40],[14,41],[20,42],[6,43],[30,44],[9,45],[11,46],[18,47],[3,48],[3,49],[27,50],[12,51],[14,52],[22,53],[30,54],[6,55],[14,56],[12,57],[2,58],[55,59],[24,60],[13,61],[40,62]]
#
#
# class Solution:
#     def findMinHeightTrees(self, n: int, edges):
#         # # 超时
#         # def bfs(i):
#         #     level = 1
#         #     stack = [i]
#         #     rem = {i}
#         #     while len(rem) < n:
#         #         tmp = []
#         #         while stack:
#         #             tmp.append(stack.pop(0))
#         #         while tmp:
#         #             stack.extend(adjacency[tmp.pop(0)])
#         #         level += 1
#         #         rem.update(stack)
#         #     return level
#         #
#         # if not edges:
#         #     return [0]
#         # adjacency = [[] for _ in range(n)]
#         # for edge in edges:
#         #     adjacency[edge[0]].append(edge[1])
#         #     adjacency[edge[1]].append(edge[0])
#         # # print(adjacency)
#         # res = [n, []]
#         # for i in range(n):
#         #     depth = bfs(i)
#         #     if depth < res[0]:
#         #         res[0] = depth
#         #         res[1] = [i]
#         #     elif depth == res[0]:
#         #         res[1].append(i)
#         #
#         # return res[1]
#
#         # # 每次删除度为1的节点
#         # if not edges:
#         #     return [0]
#         # adjacency = [[] for _ in range(n)]
#         # for edge in edges:
#         #     adjacency[edge[0]].append(edge[1])
#         #     adjacency[edge[1]].append(edge[0])
#         # # print(adjacency)
#         # degrees = [len(ad) for ad in adjacency]
#         # res = [j for j in range(n)]
#         # while len(res) > 2:
#         #     i = 0
#         #     rem = set()
#         #     while i < len(res):
#         #         if degrees[res[i]] == 1 and res[i] not in rem:
#         #             for ele in adjacency[res[i]]:
#         #                 degrees[ele] -= 1
#         #                 rem.add(ele)
#         #             res.pop(i)
#         #         else:
#         #             i += 1
#         #
#         # return res
#
#         # 更高效的写法,每次记录叶子节点
#         from collections import defaultdict
#         if not edges:
#             return [0]
#         graph = defaultdict(list)
#         for x, y in edges:
#             graph[x].append(y)
#             graph[y].append(x)
#         # 叶子节点
#         leaves = [i for i in graph if len(graph[i]) == 1]
#         while n > 2:
#             n -= len(leaves)
#             nxt_leaves = []
#             for leave in leaves:
#                 # 与叶子节点相连的点找到
#                 tmp = graph[leave].pop()
#                 # 相连的点删去这个叶子节点
#                 graph[tmp].remove(leave)
#                 if len(graph[tmp]) == 1:
#                     nxt_leaves.append(tmp)
#             leaves = nxt_leaves
#         return list(leaves)
#
#
#
#
# solve = Solution()
# print(solve.findMinHeightTrees(n, edges))
