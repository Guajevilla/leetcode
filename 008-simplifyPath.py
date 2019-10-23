import copy
################################ 71. 简化路径 #################################
# x = "/home/"                    # "/home"
# x = "/../"                      # "/"
# x = "/home//foo/"               # "/home/foo"
# x = "/a/./b/../../c/"           # "/c"
# x = "/a/../../b/../c//.//"      # "/c"
# x = "/a//b////c/d//././/.."     # "/a/b/c"
# # x = "/"
#
#
# class Solution(object):
#     def simplifyPath(self, path):
#         """
#         :type path: str
#         :rtype: str
#         """
#         res = []
#         path_lis = path.split('/')
#
#         for p in path_lis:
#             if p == '.' or p == '':
#                 continue
#             elif p == '..':
#                 if res:
#                     res.pop(-1)
#             else:
#                 res.append(p)
#
#         return '/' + '/'.join(res)
#
#
# solve = Solution()
# print(solve.simplifyPath(x))

################################ 72. 编辑距离 #################################
# word1 = "horse"
# word2 = "ros"           # 3
#
# word1 = "intention"
# word2 = "execution"     # 5
#
# # word1 = "zoologicoarchaeologist"
# # word2 = "zoogeologist"
# word1 = ''
# word2 = ''
#
#
# # 如果dp[i][j]从dp[i-1][j-1]继承来,相当于替换操作
# # 如果来自dp[i-1][j],相当于来自插入
# # 如果来自dp[i][j-1],相当于来自删除
# class Solution(object):
#     def minDistance(self, word1, word2):
#         """
#         :type word1: str
#         :type word2: str
#         :rtype: int
#         """
#         dp = [[0]*(len(word2)+1) for _ in range(len(word1)+1)]
#         for i in range(1, len(word2)+1):
#             dp[0][i] = i
#         for i in range(1, len(word1)+1):
#             dp[i][0] = i
#
#         for i in range(1, len(word1)+1):
#             for j in range(1, len(word2)+1):
#                 if word1[i-1] == word2[j-1]:
#                     dp[i][j] = dp[i-1][j-1]
#                 else:
#                     dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
#
#         return dp[-1][-1]
#
#
# solve = Solution()
# print(solve.minDistance(word1, word2))

# ################################ 73. 矩阵置零 #################################
# matrix = [
#     [1,1,1],
#     [1,0,1],
#     [1,1,1]]
# # [
# #   [1,0,1],
# #   [0,0,0],
# #   [1,0,1]
# # ]
#
# matrix = [
#   [0,1,2,0],
#   [3,4,5,2],
#   [1,3,1,5]
# ]
# # [
# #   [0,0,0,0],
# #   [0,4,5,0],
# #   [0,3,1,0]
# # ]
#
#
# class Solution(object):
#     def setZeroes(self, matrix):
#         """
#         :type matrix: List[List[int]]
#         :rtype: None Do not return anything, modify matrix in-place instead.
#         """
#         # # 这样的空间复杂度其实是不满足常数的,为 O(2*k) k表示0个数
#         # rem_i = []
#         # rem_j = []
#         # for i in range(len(matrix)):
#         #     for j in range(len(matrix[0])):
#         #         if matrix[i][j] == 0:
#         #             rem_i.append(i)
#         #             rem_j.append(j)
#         #             continue
#         #
#         # i = 0
#         # while i < len(rem_i):
#         #     matrix[rem_i[i]] = [0] * len(matrix[0])
#         #     j = 0
#         #     while j < len(matrix):
#         #         matrix[j][rem_j[i]] = 0
#         #         j += 1
#         #     i += 1
#
#
#         # O(1)空间复杂度,用矩阵第一行第一列存是否为0,而原来第一行第一列用两个flag存(这里用的是一个flag+matr[0][0])
#         flag_col = False
#         row = len(matrix[0])
#         col = len(matrix)
#         for i in range(row):
#             if matrix[i][0] == 0: flag_col = True
#             for j in range(1, col):
#                 if matrix[i][j] == 0:
#                     matrix[i][0] = matrix[0][j] = 0
#
#         for i in range(row - 1, -1, -1):
#             for j in range(col - 1, 0, -1):
#                 if matrix[i][0] == 0 or matrix[0][j] == 0:
#                     matrix[i][j] = 0
#
#             if flag_col == True: matrix[i][0] = 0
#
#
# solve = Solution()
# solve.setZeroes(matrix)
# print(matrix)

# ################################ 74. 搜索二维矩阵 #################################
# matrix = [
#   [1,   3,  5,  7],
#   [10, 11, 16, 20],
#   [23, 30, 34, 50]
# ]
# target = 3
#
# matrix = [
#   [1,   3,  5,  7],
#   [10, 11, 16, 20],
#   [23, 30, 34, 50]
# ]
# target = 10
#
# # matrix = []
#
#
# class Solution(object):
#     def searchMatrix(self, matrix, target):
#         """
#         :type matrix: List[List[int]]
#         :type target: int
#         :rtype: bool
#         """
#         # m = len(matrix)
#         # if m == 0:
#         #     return False
#         # n = len(matrix[0])
#         # if n == 0:
#         #     return False
#         # for i in range(m):
#         #     if matrix[i][0] > target:
#         #         i -= 1
#         #         break
#         # return target in matrix[i]
#
#         # 二维转一维用二分
#         if not matrix: return False
#         row = len(matrix)
#         col = len(matrix[0])
#         left = 0
#         right = row * col
#         while left < right:
#             mid = left + (right - left) // 2
#             if matrix[mid // col][mid % col] < target:
#                 left = mid + 1
#             else:
#                 right = mid
#         # print(left,right)
#         return left < row * col and matrix[left // col][left % col] == target
#
#
# solve = Solution()
# print(solve.searchMatrix(matrix, target))

# ################################ 75. 颜色分类 #################################
# # 要求仅使用常数空间的一趟扫描算法
# nums = [2,0,2,1,1,0]       # [0,0,1,1,2,2]
#
#
# class Solution(object):
#     def sortColors(self, nums):
#         """
#         :type nums: List[int]
#         :rtype: None Do not return anything, modify nums in-place instead.
#         """
#         # i = 0
#         # ind0 = 0
#         # ind1 = 0
#         # while i < len(nums):
#         #     if nums[i] == 0:
#         #         nums.insert(0, nums.pop(i))
#         #         ind0 += 1
#         #         ind1 += 1
#         #         i += 1
#         #     elif nums[i] == 1:
#         #         nums.insert(ind0, nums.pop(i))
#         #         ind1 += 1
#         #         if i >= ind0:
#         #             i += 1
#         #     else:
#         #         nums.insert(ind1, nums.pop(i))
#         #         if i >= ind1:
#         #             i += 1
#
#         # 三指针,最大的放在最右边更容易理解
#         # 对于所有 idx < p0 : nums[idx < p0] = 0
#         # curr是当前考虑元素的下标
#         p0 = curr = 0
#         # 对于所有 idx > p2 : nums[idx > p2] = 2
#         p2 = len(nums) - 1
#
#         while curr <= p2:
#             if nums[curr] == 0:
#                 nums[p0], nums[curr] = nums[curr], nums[p0]
#                 p0 += 1
#                 curr += 1
#             elif nums[curr] == 2:
#                 nums[curr], nums[p2] = nums[p2], nums[curr]
#                 p2 -= 1
#             else:
#                 curr += 1
#
#
# solve = Solution()
# solve.sortColors(nums)
# print(nums)

# ################################ 76. 最小覆盖子串 #################################
S = "ADOBECODEBANC"
T = "ABC"           # "BANC"

# S = "a"
# T = "aa"           # ""

# S = "aaaaaaaaaaaabbbbbcdd"
# T = "abcdd"

# S = "acbcad"
# T = "abcd"

# S = "bdab"
# T = "ab"

# S = "abdb"
# T = "ab"

# S = "acbbaca"
# T = "aba"       # "baca"

# S = "aaflslflsldkalskaaa"
# T = "aaa"       # "aaa"
# #
# S = "aflslflsldkalskaaa"
# T = "aa"       # "aa"
#
# S = "acbdbaab"
# T = "aabd"
#
# S = "aaabbaaba"
# T = "abbb"      # "bbaab"
#
# S = "aacbaccccaabcabbcab"
# T = "bcbbacaaab"    # "aabcabbcab"


class Solution(object):
    def minWindow(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        # rem = {}
        # ans = [0, 0, len(s)+1]
        # for ele in t:
        #     if ele in rem:
        #     # if rem.get(ele) != None:
        #         rem[ele] += 1
        #     else:
        #         rem[ele] = 1
        # l = 0
        # r = 0
        # while r < len(s):
        #     # if s[r] in rem.keys():
        #     # 这里千万不能按编译器的提示写成 if rem.get(s[r]) 这样会混淆 0 和 None; 1 和有值
        #     if rem.get(s[r]) != None:
        #         rem[s[r]] -= 1
        #     r += 1
        #     while max(rem.values()) <= 0:
        #         if r - l < ans[-1]:
        #             ans = [l, r, r - l]
        #
        #         # if s[l] not in rem.keys():
        #         if rem.get(s[l]) == None:
        #             l += 1
        #         elif rem[s[l]] < 0:
        #             rem[s[l]] += 1
        #             l += 1
        #         else:
        #             break
        #
        # if ans[-1] > len(s):
        #     return ''
        # else:
        #     return s[ans[0]: ans[1]]


        # 记录队列元素和t中元素的差值 避免每次求最大值
        from collections import defaultdict
        lookup = defaultdict(int)
        for c in t:
            lookup[c] += 1
        start = 0
        end = 0
        min_len = float("inf")
        counter = len(t)
        res = ""
        while end < len(s):
            if lookup[s[end]] > 0:
                counter -= 1
            lookup[s[end]] -= 1
            end += 1
            while counter == 0:
                if min_len > end - start:
                    min_len = end - start
                    res = s[start:end]
                if lookup[s[start]] == 0:
                    counter += 1
                lookup[s[start]] += 1
                start += 1
        return res



    # def minWindow(self, s, t):
    #     """
    #     :type s: str
    #     :type t: str
    #     :rtype: str
    #     """
    #
    #     if not t or not s:
    #         return ""
    #
    #     # Dictionary which keeps a count of all the unique characters in t.
    #     dict_t = Counter(t)
    #
    #     # Number of unique characters in t, which need to be present in the desired window.
    #     required = len(dict_t)
    #
    #     # left and right pointer
    #     l, r = 0, 0
    #
    #     # formed is used to keep track of how many unique characters in t are present in the current window in its desired frequency.
    #     # e.g. if t is "AABC" then the window must have two A's, one B and one C. Thus formed would be = 3 when all these conditions are met.
    #     formed = 0
    #
    #     # Dictionary which keeps a count of all the unique characters in the current window.
    #     window_counts = {}
    #
    #     # ans tuple of the form (window length, left, right)
    #     ans = float("inf"), None, None
    #
    #     while r < len(s):
    #
    #         # Add one character from the right to the window
    #         character = s[r]
    #         window_counts[character] = window_counts.get(character, 0) + 1
    #
    #         # If the frequency of the current character added equals to the desired count in t then increment the formed count by 1.
    #         if character in dict_t and window_counts[character] == dict_t[character]:
    #             formed += 1
    #
    #         # Try and co***act the window till the point where it ceases to be 'desirable'.
    #         while l <= r and formed == required:
    #             character = s[l]
    #
    #             # Save the smallest window until now.
    #             if r - l + 1 < ans[0]:
    #                 ans = (r - l + 1, l, r)
    #
    #             # The character at the position pointed by the `left` pointer is no longer a part of the window.
    #             window_counts[character] -= 1
    #             if character in dict_t and window_counts[character] < dict_t[character]:
    #                 formed -= 1
    #
    #             # Move the left pointer ahead, this would help to look for a new window.
    #             l += 1
    #
    #         # Keep expanding the window once we are done co***acting.
    #         r += 1
    #     return "" if ans[0] == float("inf") else s[ans[1] : ans[2] + 1]


solve = Solution()
print(solve.minWindow(S, T))

# ################################ 77. 组合 #################################
# n = 4
# k = 2
# # [
# #   [2,4],
# #   [3,4],
# #   [2,3],
# #   [1,2],
# #   [1,3],
# #   [1,4],
# # ]
#
#
# class Solution(object):
#     def combine(self, n, k):
#         """
#         :type n: int
#         :type k: int
#         :rtype: List[List[int]]
#         """
#         res = []
#
#         def backtrack(i, tmp):
#             if len(tmp) == k:
#                 res.append(tmp)
#                 return
#             for j in range(i, n+1):
#                 backtrack(j+1, tmp+[j])
#
#         backtrack(1, [])
#         return res
#
#
# solve = Solution()
# print(solve.combine(n, k))

# # ################################ 78. 子集 #################################
# nums = [1,2,3]
# # [
# #   [3],
# #   [1],
# #   [2],
# #   [1,2,3],
# #   [1,3],
# #   [2,3],
# #   [1,2],
# #   []
# # ]
#
#
# class Solution(object):
#     def subsets(self, nums):
#         """
#         :type nums: List[int]
#         :rtype: List[List[int]]
#         """
#         def backtrack(nums):
#             if not nums:
#                 return
#             tmp = copy.deepcopy(res[:])
#             for i in range(len(tmp)):
#                 tmp[i].append(nums[0])
#             res.extend(tmp)
#             backtrack(nums[1:])
#
#         res = [[]]
#         backtrack(nums)
#         return res
#
#         # res = []
#         # n = len(nums)
#         #
#         # def backtrack(i, tmp):
#         #     res.append(tmp)
#         #     for j in range(i, n):
#         #         backtrack(j + 1, tmp + [nums[j]])
#         #
#         # backtrack(0, [])
#         # return res
#
#
#         # # 迭代
#         # res = [[]]
#         # for i in nums:
#         #     res = res + [[i] + num for num in res]
#         # return res
#
#
# solve = Solution()
# print(solve.subsets(nums))

# # ################################ 79. 单词搜索 #################################
# import copy
# board =[
#   ['A','B','C','E'],
#   ['S','F','C','S'],
#   ['A','D','E','E']
# ]
#
# word = "ABCCED" # true.
# # word = "SEE"    # true.
# # word = "ABCB"   # false.
#
# # board = [["A","B","C","E"],["S","F","E","S"],["A","D","E","E"]]
# # word = "ABCESEEEFS"
#
# board = [["a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a"],
#          ["a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a"],
#          ["a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a"],
#          ["a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a"],
#          ["a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a"],
#          ["a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a"],
#          ["a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a"],
#          ["a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a"],
#          ["a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a"],
#          ["a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a"],
#          ["a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a"],
#          ["a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a"],
#          ["a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a"],
#          ["a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a"],
#          ["a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a"],
#          ["a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a"],
#          ["a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a"],
#          ["a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a"],
#          ["a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a"],
#          ["a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a"],
#          ["a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a"],
#          ["a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a"],
#          ["a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a"],
#          ["a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a"],
#          ["a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a"],
#          ["a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a"],
#          ["a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a"],
#          ["a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a"],
#          ["a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a"],
#          ["a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","b"]]
# word = "baaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
#
#
# class Solution(object):
#     # TODO 返回一个i,j变成1的board，但不能改变原board
#     def matrixset(self, board, i, j):
#         board1 = board[:]
#         board1[i] = board[i][:j] + [1] + board[i][j+1:]
#         return board1
#
#     def exist(self, board, word):
#         """
#         :type board: List[List[str]]
#         :type word: str
#         :rtype: bool
#         """
#         # 存在list里还要in判断,耗时,改成占空间的等大矩阵
#         # m = len(board)
#         # n = len(board[0])
#         #
#         # def find_ele(word, ii, jj, rem):
#         #     tmp = 0
#         #     if word == '':
#         #         return True
#         #     for x, y in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
#         #         tmp_i = x + ii
#         #         tmp_j = y + jj
#         #         if 0 <= tmp_i < m and 0 <= tmp_j < n and [tmp_i, tmp_j] not in rem and board[tmp_i][tmp_j] == word[0]:
#         #             # rem.append([tmp_i, tmp_j])
#         #             tmp += find_ele(word[1:], tmp_i, tmp_j, rem+[[tmp_i, tmp_j]])
#         #
#         #     return tmp > 0
#         #
#         # tmp = 0
#         # for i in range(m):
#         #     for j in range(n):
#         #         if word[0] == board[i][j]:
#         #             tmp += find_ele(word[1:], i, j, [[i,j]])
#         # return tmp > 0
#
#         m = len(board)
#         n = len(board[0])
#
#         def find_ele(word, ii, jj, rem):
#             tmp = 0
#             if word == '':
#                 return True
#             for x, y in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
#                 tmp_i = x + ii
#                 tmp_j = y + jj
#                 if 0 <= tmp_i < m and 0 <= tmp_j < n and rem[tmp_i][tmp_j] != 1 and board[tmp_i][tmp_j] == word[0]:
#                     tmp += find_ele(word[1:], tmp_i, tmp_j, self.matrixset(rem, tmp_i, tmp_j))
#                     # 剪枝,不然容易超时
#                     if tmp > 0:
#                         return True
#
#             return tmp > 0
#
#         tmp = 0
#         rem = [[0]*n for _ in range(m)]
#         for i in range(m):
#             for j in range(n):
#                 if word[0] == board[i][j]:
#                     tmp += find_ele(word[1:], i, j, self.matrixset(rem, i, j))
#         return tmp > 0
#
#
# class Solution1(object):
#     def exist(self, board, word):
#         row = len(board)
#         col = len(board[0])
#
#         def helper(i, j, k, visited):
#             if k == len(word):
#                 return True
#             for x, y in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
#                 tmp_i = x + i
#                 tmp_j = y + j
#                 if 0 <= tmp_i < row and 0 <= tmp_j < col and (tmp_i, tmp_j) not in visited \
#                         and board[tmp_i][tmp_j] == word[k]:
#                     visited.add((tmp_i, tmp_j))
#                     if helper(tmp_i, tmp_j, k + 1, visited):
#                         return True
#                     visited.remove((tmp_i, tmp_j))  # 回溯
#             return False
#
#         for i in range(row):
#             for j in range(col):
#                 if board[i][j] == word[0] and helper(i, j, 1, {(i, j)}):
#                     return True
#         return False
#
#
# solve = Solution()
# print(solve.exist(board, word))


############################## 80. 删除排序数组中的重复项 II #############################
# nums = [1,1,1,2,2,3]        # 5
# nums = [0,0,1,1,1,1,2,3,3]  # 7
# # nums = []        # 5
#
#
# class Solution(object):
#     def removeDuplicates(self, nums):
#         """
#         :type nums: List[int]
#         :rtype: int
#         """
#         # i = 2
#         # while i < len(nums):
#         #     if nums[i] == nums[i-2]:
#         #         nums.pop(i)
#         #     else:
#         #         i += 1
#         # return len(nums)
#
#         # 用i保存当前有效index
#         i = 0
#         for n in nums:
#             if i < 2 or n != nums[i - 2]:
#                 nums[i] = n
#                 i += 1
#         return i
#
#
# solve = Solution()
# print(solve.removeDuplicates(nums))
