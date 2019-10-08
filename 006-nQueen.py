# # ################################ 51. N皇后 #################################
# N = 2 无解
# N = 3 无解
# N = 4 2解
N = 4
# [
#  [".Q..",  // 解法 1
#   "...Q",
#   "Q...",
#   "..Q."],
#
#  ["..Q.",  // 解法 2
#   "Q...",
#   "...Q",
#   ".Q.."]
# ]


# class Solution(object):
#     def deleteQ(self, board, i, j):
#         n = len(board)
#         board[i] = ['.'] * j + ['Q'] + ['.'] * (n - j - 1)
#         for ii in range(n):
#             if ii != i:
#                 board[ii][j] = '.'
#             for jj in range(n):
#                 if abs(i-ii) == abs(j-jj) and ii != i and jj != j:
#                     board[ii][jj] = '.'
#         return board
#
#     def tryOnce(self, board1, i, j, cnt, res):
#         import copy
#         board = copy.deepcopy(board1)
#         n = len(board)
#         for ii in range(i, n):
#             for jj in range(n):
#                 if ii == i and jj < j:
#                     continue
#                 if board[ii][jj] == 'Q':
#                     if jj != n - 1 and 'Q' in board[ii][(jj+1):]:
#                         self.tryOnce(board, ii, jj+1+board[ii][(jj+1):].index('Q'), cnt, res)
#                     board = self.deleteQ(board, ii, jj)
#                     cnt += 1
#                     continue
#
#         if cnt == n:
#             tmp_list = []
#             for c in range(n):
#                 tmp = "".join(board[c])
#                 tmp_list.append(tmp)
#             res.append(tmp_list)
#
#     def solveNQueens(self, n):
#         """
#         :type n: int
#         :rtype: List[List[str]]
#         """
#         res = []
#         board = [["Q"] * n for _ in range(n)]
#         self.tryOnce(board, 0, 0, 0, res)
#
#         return res

"""
引用全局变量,不需要golbal声明.修改全局变量,需要使用global声明
特别地.列表.字典等如果只是修改其中元素的值.可以直接使用全局变量.不需要global声明.
"""
# queen 存每一行Queen所在的索引,所以天然能保证不在同一行
class Solution1(object):
    def solveNQueens(self, n):
        def DFS(queens, xy_dif, xy_sum):
            p = len(queens)
            if p == n:
                result.append(queens)
                return
            for q in range(n):
                # 保证不在同一列(not in queens) 不在对角线(p-q not in xy_dif and p+q not in xy_sum)
                if q not in queens and p-q not in xy_dif and p+q not in xy_sum:
                    DFS(queens+[q], xy_dif+[p-q], xy_sum+[p+q])
        result = []
        DFS([], [], [])
        return [["."*i + "Q" + "."*(n-i-1) for i in sol] for sol in result]


N = 8
# solve = Solution()
solve1 = Solution1()
# print(solve.solveNQueens(N))
# print(len(solve.solveNQueens(N)))
print(solve1.solveNQueens(N))
print(len(solve1.solveNQueens(N)))

# n = 4
# board = [['Q'] * n for _ in range(n)]
# solve.deleteQ(board, 0, 1)
# print(board)

# # ################################ 52. N皇后 II #################################
# N = 8
#
#
# class Solution(object):
#     def deleteQ(self, board, i, j):
#         n = len(board)
#         board[i] = ['.'] * j + ['Q'] + ['.'] * (n - j - 1)
#         for ii in range(n):
#             if ii != i:
#                 board[ii][j] = '.'
#             for jj in range(n):
#                 if abs(i-ii) == abs(j-jj) and ii != i and jj != j:
#                     board[ii][jj] = '.'
#         return board
#
#     def tryOnce(self, board1, i, j, cnt):
#         num = 0
#         import copy
#         board = copy.deepcopy(board1)
#         n = len(board1)
#         for ii in range(i, n):
#             for jj in range(n):
#                 if ii == i and jj < j:
#                     continue
#                 if board[ii][jj] == 'Q':
#                     if jj != n - 1 and 'Q' in board[ii][(jj+1):]:
#                         num += self.tryOnce(board, ii, jj+1+board[ii][(jj+1):].index('Q'), cnt)
#                     board = self.deleteQ(board, ii, jj)
#                     cnt += 1
#                     continue
#         # print(cnt)
#         if cnt == n:
#             # print(board)
#             num += 1
#         return num
#
#     def totalNQueens(self, n):
#         """
#         :type n: int
#         :rtype: int
#         """
#         cnt = 0
#         board = [["Q"] * n for _ in range(n)]
#         cnt += self.tryOnce(board, 0, 0, 0)
#
#         return cnt


class Solution1(object):
    def totalNQueens(self, n):
        def DFS(queens, xy_dif, xy_sum):
            p = len(queens)
            if p == n:
                cnt[0] += 1
                return
            for q in range(n):
                if q not in queens and p-q not in xy_dif and p+q not in xy_sum:
                    DFS(queens+[q], xy_dif+[p-q], xy_sum+[p+q])
        cnt = [0]
        DFS([], [], [])
        return cnt[0]


# solve = Solution()
# print(solve.totalNQueens(N))
solve = Solution1()
# print(len(solve.totalNQueens(N)))
print(solve.totalNQueens(N))

########################## 53. 最大子序和 #################################
# nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]  # 6
#
#
# # nums = []
#
#
# class Solution(object):
#     def maxSubArray(self, nums):
#         """
#         :type nums: List[int]
#         :rtype: int
#         """
#         max_num = nums[0]
#         for i in range(1, len(nums)):
#             nums[i] = nums[i] + max(nums[i - 1], 0)
#             if nums[i] > max_num:
#                 max_num = nums[i]
#
#         return max_num
#
#         # tmp = nums[0]
#         # max_ = tmp
#         # n = len(nums)
#         # for i in range(1, n):
#         #     # 当当前序列加上此时的元素的值大于tmp的值，说明最大序列和可能出现在后续序列中，记录此时的最大值
#         #     if tmp + nums[i] > nums[i]:
#         #         max_ = max(max_, tmp + nums[i])
#         #         tmp = tmp + nums[i]
#         #     else:
#         #         # 当tmp(当前和)小于下一个元素时，当前最长序列到此为止。以该元素为起点继续找最大子序列,
#         #         # 并记录此时的最大值
#         #         max_ = max(max_, tmp, tmp + nums[i], nums[i])
#         #         tmp = nums[i]
#         # return max_
#
#
# solve = Solution()
# print(solve.maxSubArray(nums))

########################### 54. 螺旋矩阵 #################################
# m = [[ 1, 2, 3 ],
#      [ 4, 5, 6 ],
#      [ 7, 8, 9 ]]   # [1,2,3,6,9,8,7,4,5]
#
# # m = [[1],[2],[3]]
#
# m = [
#   [1, 2, 3, 4],
#   [5, 6, 7, 8],
#   [9,10,11,12]
# ]       # [1,2,3,4,8,12,11,10,9,5,6,7]
#
# m = [[]]
#
#
# # 另外比较巧妙的方法是每次弹出第一行，然后将剩余矩阵逆时针旋转90度
# class Solution(object):
#     def outerCircle(self, mat):
#         res = []
#         i = 0
#         if len(mat[0]) == 1:
#             while i < len(mat):
#                 res.append(mat[i].pop(0))
#                 if not mat[i]:
#                     mat.pop(i)
#                     continue
#                 i += 1
#             return res
#         elif len(mat) == 1:
#             return mat.pop(0)
#         res = mat.pop(0)
#         tmp = []
#         while i < len(mat)-1:
#             res.append(mat[i].pop(-1))
#             tmp.append(mat[i].pop(0))
#             if not mat[i]:
#                 mat.pop(i)
#                 continue
#             i += 1
#         tmp.extend(mat.pop(-1))
#         res.extend(tmp[::-1])
#         return res
#
#     def spiralOrder(self, matrix):
#         """
#         :type matrix: List[List[int]]
#         :rtype: List[int]
#         """
#         res = []
#         while len(matrix):
#             res.extend(self.outerCircle(matrix))
#
#         return res
#
#
# solve = Solution()
# print(solve.spiralOrder(m))

# ########################### 55. 跳跃游戏 #################################
# num = [2,3,1,1,4]   # True
# # num = [3,2,1,0,4]   # False
# # num = [2,3,1,1,4]       # 2
# # num = [10,9,8,7,6,5,4,3,2,1,1,0,0]
# num = [5,2,1,1,1,3,4]       # 2
# num = [7,0,9,6,9,6,1,7,9,0,1,2,9,0,3]       # 2
# # num = [0,1]
#
#
# class Solution(object):
#     def canJump(self, nums):
#         """
#         :type nums: List[int]
#         :rtype: bool
#         """
#         # ind = 1
#         # rem = -1
#         # while ind <= len(nums) - 1:
#         #     tmp_ind = -1
#         #     tmp_max = 0
#         #     if nums[ind-1] == 0:
#         #         return False
#         #     for i, num in enumerate(nums[ind: (ind+nums[ind-1])]):
#         #         if i + num > tmp_max:
#         #             tmp_max = i + num
#         #             tmp_ind = i
#         #     if ind + tmp_max >= len(nums) - 1:
#         #         return True
#         #     ind += tmp_ind + 1
#         #     if ind == rem:
#         #         return False
#         #     rem = ind
#         # return True
#
#
# solve = Solution()
# print(solve.canJump(num))

# ########################### 56. 合并区间 #################################
# intervall = [[1,3],[2,6],[8,10],[15,18]]      # [[1,6],[8,10],[15,18]]
# # intervall = [[1,4],[4,5]]        # [[1,5]]
# # intervall = [[1,10],[2,6],[8,10],[15,18]]      # [[1,10],[15,18]]
# intervall = [[2,6],[1,3],[8,10],[15,18]]      # [[1,6],[8,10],[15,18]]
# intervall = [[2,6]]
#
#
# class Solution(object):
#     def merge(self, intervals):
#         """
#         :type intervals: List[List[int]]
#         :rtype: List[List[int]]
#         """
#         l_board = []
#         r_board = []
#         for interval in intervals:
#             l_board.append(interval[0])
#             r_board.append(interval[1])
#         l = 1
#         r = 0
#         l_board.sort()
#         r_board.sort()
#         while l < len(l_board) and r < len(r_board):
#             if l_board[l] <= r_board[r]:
#                 if r_board[r] <= r_board[r+1]:
#                     l_board.pop(l)
#                     r_board.pop(r)
#                 else:
#                     l_board.pop(l)
#                     r_board.pop(r+1)
#             else:
#                 l += 1
#                 r += 1
#         res = []
#         while len(l_board):
#             res.append([l_board.pop(0), r_board.pop(0)])
#
#         return res
#
#
# solve = Solution()
# print(solve.merge(intervall))

# ########################### 57. 插入区间 #################################
# intervals = [[1,3],[6,9]]
# newInterval = [2,5]     # [[1,5],[6,9]]
#
# # intervals = [[1,2],[3,5],[6,7],[8,10],[12,16]]
# # newInterval = [4,8]     # [[1,2],[3,10],[12,16]]
#
# intervals = [[1,5]]
# newInterval = [1,7]
#
#
# class Solution(object):
#     def insert(self, intervals, newInterval):
#         """
#         :type intervals: List[List[int]]
#         :type newInterval: List[int]
#         :rtype: List[List[int]]
#         """
#         l_board = []
#         r_board = []
#         l_flag = 1
#         r_flag = 1
#         for interval in intervals:
#             if interval[0] >= newInterval[0] and l_flag:
#                 l_board.append(newInterval[0])
#                 l_flag = 0
#             if interval[1] >= newInterval[1] and r_flag:
#                 r_board.append(newInterval[1])
#                 r_flag = 0
#
#             l_board.append(interval[0])
#             r_board.append(interval[1])
#
#         if l_flag:
#             l_board.append(newInterval[0])
#         if r_flag:
#             r_board.append(newInterval[1])
#         l = 1
#         r = 0
#         while l < len(l_board) and r < len(r_board):
#             if l_board[l] <= r_board[r]:
#                 if r_board[r] <= r_board[r+1]:
#                     l_board.pop(l)
#                     r_board.pop(r)
#                 else:
#                     l_board.pop(l)
#                     r_board.pop(r+1)
#             else:
#                 l += 1
#                 r += 1
#
#         res = []
#         while len(l_board):
#             res.append([l_board.pop(0), r_board.pop(0)])
#
#         return res
#
#
# solve = Solution()
# print(solve.insert(intervals, newInterval))

########################### 58. 最后一个单词的长度 #################################
#
#
# class Solution(object):
#     def lengthOfLastWord(self, s):
#         """
#         :type s: str
#         :rtype: int
#         """
#         s = s.rstrip()
#         if not s:
#             return 0
#         ind = s[::-1].find(' ')
#         # print(ind)
#         if ind != -1:
#             return ind
#         else:
#             return len(s)


# # ################################ 59. 螺旋矩阵 II #################################
# N = 3
# # [
# #  [ 1, 2, 3 ],
# #  [ 8, 9, 4 ],
# #  [ 7, 6, 5 ]
# # ]
#
#
# class Solution(object):
#     def inserOuter(self, n, tmp, cnt, lis):
#         circle = (tmp - n) // 2
#         if circle == 0:
#             end = 4 * (n - 1)
#             for i in range(n):
#                 lis[0].append(1+i)
#                 lis[-1].insert(0, 2*n-1+i)
#                 if 0 < i < n-1:
#                     lis[i].append(end-i+1)
#                     lis[i].append(n+i)
#         else:
#             begin = cnt + 1
#             end = 4 * (n - 1) + begin - 1
#             for i in range(n):
#                 lis[circle].insert(-circle, begin+i)
#                 lis[n-1+circle].insert(circle, begin+2*n-2+i)
#                 if 0 < i < n-1:
#                     lis[circle+i].insert(-circle, begin+n+i-1)
#                     lis[circle+i].insert(circle, end-i+1)
#
#         return lis
#
#     def generateMatrix(self, n):
#         """
#         :type n: int
#         :rtype: List[List[int]]
#         """
#         tmp = n
#         res = [[] for _ in range(n)]
#         cnt = 0
#         while n > 1:
#             res = self.inserOuter(n, tmp, cnt, res)
#             cnt += 4*n-4
#             n -= 2
#
#         if n == 1:
#             res[tmp//2].insert(tmp//2, tmp**2)
#         return res
#
#     # # 直接判断边界顺序填入
#     # def generateMatrix(self, n):
#     #     l, r, t, b = 0, n - 1, 0, n - 1
#     #     mat = [[0 for _ in range(n)] for _ in range(n)]
#     #     num, tar = 1, n * n
#     #     while num <= tar:
#     #         for i in range(l, r + 1): # left to right
#     #             mat[t][i] = num
#     #             num += 1
#     #         t += 1
#     #         for i in range(t, b + 1): # top to bottom
#     #             mat[i][r] = num
#     #             num += 1
#     #         r -= 1
#     #         for i in range(r, l - 1, -1): # right to left
#     #             mat[b][i] = num
#     #             num += 1
#     #         b -= 1
#     #         for i in range(b, t - 1, -1): # bottom to top
#     #             mat[i][l] = num
#     #             num += 1
#     #         l += 1
#     #     return mat
#
#
# solve = Solution()
# print(solve.generateMatrix(N))
# # print(solve.inserOuter(5,5,[]))
# # print(solve.inserOuter(3,5,[[1,2,3,4,5],[16,6],[15,7],[14,8],[13,12,11,10,9]]))
# # print(solve.inserOuter(2,4,[[1,2,3,4],[12,5],[11,6],[10,9,8,7]]))

# # ################################ 60. 第k个排列 #################################
# n = 3
# k = 3    # "213"
#
# n = 4
# k = 9    # "2314"
#
#
# class Solution(object):
#     def getPermutation(self, n, k):
#         """
#         :type n: int
#         :type k: int
#         :rtype: str
#         """
#         # from math import factorial as fact
#         # res = ''
#         # rem = [i+1 for i in range(n)]
#         # while rem:
#         #     ind = (k-1) // fact(n - 1)
#         #     res += str(rem.pop(ind))
#         #     k = k % fact(n-1)
#         #     n -= 1
#
#         # 减少求fact次数以加快运行速度
#         from math import factorial as fact
#         res = ''
#         rem = [i+1 for i in range(n)]
#         div_fact = fact(n - 1)
#         while True:
#             ind = (k-1) // div_fact
#             res += str(rem.pop(ind))
#
#             k = k % div_fact
#             n -= 1
#             if n == 0:
#                 break
#             div_fact = div_fact // n
#
#         return res
#
#
# k=4
# solve = Solution()
# print(solve.getPermutation(n, k))
