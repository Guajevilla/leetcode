# # ################################ 61. 旋转链表 #################################
#
#
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None
#
#
# def printList(l: ListNode):
#     while l:
#         print("%d, " %(l.val), end = '')
#         l = l.next
#     print('')
#
#
# x = ListNode(1)
# x.next = ListNode(2)
# x.next.next = ListNode(3)
# x.next.next.next = ListNode(4)
# x.next.next.next.next = ListNode(5)
# k = 2       # 4->5->1->2->3->NULL
#
# x = ListNode(0)
# x.next = ListNode(1)
# x.next.next = ListNode(2)
# k = 4       # 2->0->1->NULL
# #
# # x = ListNode(0)
# # k = 1
#
#
# class Solution(object):
#     def rotateRight(self, head, k):
#         """
#         :type head: ListNode
#         :type k: int
#         :rtype: ListNode
#         """
#         # 成环的思路
#         if not head or not head.next:
#             return head
#         length = 1
#         p = head
#         while p.next:
#             length += 1
#             p = p.next
#         p.next = head
#
#         for _ in range(length - k % length):
#             p = head
#             head = head.next
#         p.next = None
#         return head
#
#         # 不成环，直接找到头，然后将头尾连起来
#         # if not head or not head.next: return head
#         # # 链表个数
#         # num = 0
#         # p = head
#         # while p:
#         #     num += 1
#         #     p = p.next
#         # k = num - k % num
#         # p = head
#         # # 找前一段链表
#         # while k > 1:
#         #     p = p.next
#         #     k -= 1
#         # head1 = p.next
#         # if not head1: return head
#         # # 前一段链表最后至空
#         # p.next = None
#         # p = head1
#         # # 后一段链表和前一段链表连接起来
#         # while p.next:
#         #     p = p.next
#         # p.next = head
#         # return head1
#
#
# solve = Solution()
# printList(solve.rotateRight(x, k))

# # ################################ 62. 不同路径 #################################
# m = 3
# n = 2   # 3
#
# m = 7
# n = 5   # 28
#
# m = 23
# n = 12   # 28
#
#
# # 递归超时了
# class Solution(object):
#     # def position(self, m, n, i, j):
#     #     cnt = 0
#     #     tmp = 0
#     #     if i < m - 1:
#     #         cnt += self.position(m, n, i+1, j)
#     #         tmp += 1
#     #     if j < n - 1:
#     #         cnt += self.position(m, n, i, j+1)
#     #         tmp += 1
#     #     if tmp == 2:
#     #         cnt += 1
#     #     return cnt
#     #
#     # def uniquePaths(self, m, n):
#     #     """
#     #     :type m: int
#     #     :type n: int
#     #     :rtype: int
#     #     """
#     #     return self.position(m, n, 0, 0) + 1
#
#
#     # 动态规划
#     def uniquePaths(self, m, n):
#         """
#         :type m: int
#         :type n: int
#         :rtype: int
#         """
#         dp = [[1]*n for _ in range(m)]
#         for i in range(1, m):
#             for j in range(1, n):
#                 dp[i][j] = dp[i-1][j] + dp[i][j-1]
#         return dp[m-1][n-1]
#
#
#         # # 直接数学解法 机器人一定会走m+n-2步，即从m+n-2中挑出m-1步向下走 即C（（m+n-2），（m-1））
#         # return int(math.factorial(m + n - 2) / math.factorial(m - 1) / math.factorial(n - 1))
#
#
# solve = Solution()
# print(solve.uniquePaths(m, n))

# # ################################ 63. 不同路径 II #################################
# obstacleGrid = [
#   [0,0,0],
#   [0,1,0],
#   [0,0,0]
# ]           # 2
# obstacleGrid = [[0,0],[1,0]]
# # obstacleGrid = [[0,0,0,1,0],[0,0,0,0,0],[0,0,1,0,0]]
# # obstacleGrid = [[0]]
#
#
# class Solution(object):
#     def uniquePathsWithObstacles(self, obstacleGrid):
#         """
#         :type obstacleGrid: List[List[int]]
#         :rtype: int
#         """
#         m = len(obstacleGrid)
#         n = len(obstacleGrid[0])
#         dp = [[0]*n for _ in range(m)]
#         for i in range(m):
#             if obstacleGrid[i][0] == 0:
#                 dp[i][0] = 1
#             else:
#                 break
#         for i in range(n):
#             if obstacleGrid[0][i] == 0:
#                 dp[0][i] = 1
#             else:
#                 break
#
#         for i in range(1, m):
#             for j in range(1, n):
#                 if obstacleGrid[i][j] == 1:
#                     dp[i][j] = 0
#                 else:
#                     dp[i][j] = dp[i-1][j] + dp[i][j-1]
#         return dp[m-1][n-1]
#
#
# solve = Solution()
# print(solve.uniquePathsWithObstacles(obstacleGrid))

# ################################ 64. 最小路径和 #################################
# grid = [
#   [1,3,1],
#   [1,5,1],
#   [4,2,1]
# ]           # 7
#
#
# class Solution(object):
#     def minPathSum(self, grid):
#         """
#         :type grid: List[List[int]]
#         :rtype: int
#         """
#         if not grid:
#             return 0
#         m = len(grid)
#         n = len(grid[0])
#         for i in range(1, m):
#             grid[i][0] += grid[i-1][0]
#         for i in range(1, n):
#             grid[0][i] += grid[0][i-1]
#
#         for i in range(1, m):
#             for j in range(1, n):
#                 grid[i][j] += min(grid[i-1][j], grid[i][j-1])
#
#         return grid[-1][-1]
#
#
# solve = Solution()
# print(solve.minPathSum(grid))

# ################################ 65. 有效数字 #################################
# s = "0"         # true
# s = " 0.1 "     # true
# s = "abc"       # false
# s = "1 a"       # false
# s = "2e10"      # true
# s = " -90e3   " # true
# s = " 1e"       # false
# s = "e3"        # false
# s = " 6e-1"     # true
# s = " 99e2.5 "  # false
# s = "53.5e93"   # true
# s = " --6 "     # false
# s = "-+3"       # false
# s = "95a54e53"  # false
# s = "1 e 1"       # false
# s = "..."
# s = " -."
#
#
# class Solution(object):
#     def isNumber(self, s):
#         """
#         :type s: str
#         :rtype: bool
#         """
#         s = s.strip()
#         dot_seen = 0
#         e_seen = 0
#         num_seen = 0
#         for i, a in enumerate(s):
#             if a.isdigit():
#                 num_seen = 1
#             elif a == ".":
#                 if e_seen or dot_seen:
#                     return False
#                 dot_seen = 1
#             elif a == "e":
#                 if e_seen or not num_seen:
#                     return False
#                 num_seen = 0
#                 e_seen = 1
#             elif a in "+-":
#                 if i > 0 and s[i - 1] != "e":
#                     return False
#             else:
#                 return False
#         return num_seen
#
#
# solve = Solution()
# print(solve.isNumber(s))

# ################################ 66. 加一 #################################
# num = [1,2,3]       # [1,2,4]
# # num = [4,3,2,1]     # [4,3,2,2]
# num = [9,9]       # [1,2,4]
# # num = [0]       # [1,2,4]
#
#
# class Solution(object):
#     def plusOne(self, digits):
#         """
#         :type digits: List[int]
#         :rtype: List[int]
#         """
#         i = len(digits) - 1
#         rem = 0
#         first_flag = 1
#         while i >= 0:
#             digits[i] = first_flag + digits[i] + rem
#             first_flag = 0
#             if digits[i] == 10:
#                 rem = 1
#                 digits[i] = 0
#             else:
#                 rem = 0
#                 break
#             i -= 1
#         if rem == 1:
#             digits.insert(0, 1)
#         return digits
#
#
# solve = Solution()
# print(solve.plusOne(num))

# ################################ 67. 二进制求和 #################################
# a = "11"
# b = "1"     # "100"
#
# # a = "1010"
# # b = "1011"  # "10101"
#
# a = "1"
# b = "11111"     # "100"
#
#
# class Solution(object):
#     def addBinary(self, a, b):
#         """
#         :type a: str
#         :type b: str
#         :rtype: str
#         """
#         res = ''
#         rem = 0
#         i = len(a) - 1
#         j = len(b) - 1
#         while i >= 0 and j >= 0:
#             tmp = int(a[i]) + int(b[j]) + rem
#             if tmp >= 2:
#                 res = str(tmp % 2) + res
#                 rem = 1
#             else:
#                 res = str(tmp) + res
#                 rem = 0
#             i -= 1
#             j -= 1
#
#         while i >= 0:
#             tmp = int(a[i]) + rem
#             if tmp >= 2:
#                 res = str(tmp % 2) + res
#                 rem = 1
#             else:
#                 res = str(tmp) + res
#                 rem = 0
#             i -= 1
#         while j >= 0:
#             tmp = int(b[j]) + rem
#             if tmp >= 2:
#                 res = str(tmp % 2) + res
#                 rem = 1
#             else:
#                 res = str(tmp) + res
#                 rem = 0
#             j -= 1
#         if rem == 1:
#             res = str(1) + res
#         return res
#
#         # # 简短写法
#         # res = ""
#         # carry = 0
#         # i = len(a) - 1
#         # j = len(b) - 1
#         # while i >= 0 or j >= 0 or carry:
#         #     tmp1 = int(a[i]) if i >= 0 else 0
#         #     tmp2 = int(b[j]) if j >= 0 else 0
#         #     carry, t = divmod(tmp1 + tmp2 + carry, 2)
#         #     res = str(t) + res
#         #     i -= 1
#         #     j -= 1
#         # return res
#
#
# solve = Solution()
# print(solve.addBinary(a, b))

# ################################ 68. 文本左右对齐 #################################
# words = ["This", "is", "an", "example", "of", "text", "justification.."]
# maxWidth = 16
# # [
# #    "This    is    an",
# #    "example  of text",
# #    "justification.  "
# # ]
#
# words = ["What","must","be","acknowledgment","shall","be"]
# maxWidth = 16
# # [
# #   "What   must   be",
# #   "acknowledgment  ",
# #   "shall be        "
# # ]
#
#
# class Solution(object):
#     def fullJustify(self, words, maxWidth):
#         """
#         :type words: List[str]
#         :type maxWidth: int
#         :rtype: List[str]
#         """
#         res = []
#         length = []
#         for word in words:
#             length.append(len(word))
#         word_length = 0
#         space_length = 0
#         space_add = 0
#         i = 0
#         for l in length:
#             if word_length + l + space_length <= maxWidth:
#                 word_length += l
#                 space_length += 1
#             else:
#                 if space_length == 1:
#                     space_num = maxWidth - word_length
#                 else:
#                     space_num = (maxWidth - word_length) // (space_length - 1)
#                     space_add = (maxWidth - word_length) % (space_length - 1)
#                 tmp = ''
#                 tmp_length = i + space_length
#                 while i < tmp_length:
#                     tmp = tmp + words[i]
#                     if i != tmp_length - 1 or space_length == 1:
#                         tmp = tmp + ' ' * space_num
#                         if space_add != 0:
#                             tmp = tmp + ' '
#                             space_add -= 1
#                     i += 1
#                 res.append(tmp)
#                 word_length = l
#                 space_length = 1
#
#         tmp = ''
#         while i < len(words):
#             tmp = tmp + words[i]
#             if i != len(words) - 1:
#                 tmp = tmp + ' '
#             i += 1
#         tmp += ' ' * (maxWidth-len(tmp))
#         res.append(tmp)
#         return res
#
#
# solve = Solution()
# print(solve.fullJustify(words, maxWidth))

# ################################ 69. x 的平方根 #################################
# x = 4
# x = 8       # 2
#
#
# class Solution(object):
#     def mySqrt(self, x):
#         """
#         :type x: int
#         :rtype: int
#         """
#         # # 二分法
#         # left = 0
#         # right = x // 2
#         # while left < right:
#         #     mid = left + (right - left) // 2
#         #     tmp = mid * mid
#         #     if tmp == x:
#         #         return mid
#         #     elif tmp < x:
#         #         left = mid + 1
#         #     else:
#         #         right = mid - 1
#         # return right
#
#
#         # 牛顿迭代
#         if x <= 1:
#             return x
#         r = x
#         while r > x / r:
#             r = (r + x / r) // 2
#         return int(r)
#
#
# solve = Solution()
# print(solve.mySqrt(x))

# ################################ 70. 爬楼梯 #################################
n = 2       # 2
# n = 15       # 987


class Solution(object):
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n == 1:
            return 1
        dp = [0] * n
        dp[0] = 1
        dp[1] = 2
        for i in range(2, n):
            dp[i] = dp[i-1] + dp[i-2]
        return dp[-1]


solve = Solution()
print(solve.climbStairs(n))
