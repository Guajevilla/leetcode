import json
# ############################### 321. 拼接最大数 ################################
# nums1 = [3, 4, 6, 5]
# nums2 = [9, 1, 2, 5, 8, 3]
# k = 5               # [9, 8, 6, 5, 3]
#
# nums1 = [6, 7]
# nums2 = [6, 0, 4]
# k = 5               # [6, 7, 6, 0, 4]
#
# nums1 = [3, 9]
# nums2 = [8, 9]
# k = 3               # [9, 8, 9]
#
#
# class Solution:
#     def maxNumber(self, nums1, nums2, k):
#         def findKmax(nums, k):
#             if k == 0:
#                 return []
#             elif k > len(nums):
#                 return -1
#             res = []
#             remain = len(nums) - k
#             for num in nums:
#                 while res and remain and num > res[-1]:
#                     res.pop(-1)
#                     remain -= 1
#                 res.append(num)
#             return res[:k]
#
#         def merge(lis1, lis2):
#             return [max(lis1, lis2).pop(0) for _ in range(k)]
#             # res = []
#             # i = 0
#             # j = 0
#             # while i < len(lis1) and j < len(lis2):
#             #     if lis1[i] > lis2[j]:
#             #         res.append(lis1[i])
#             #         i += 1
#             #     else:
#             #         res.append(lis2[j])
#             #         j += 1
#             # res.extend(lis1[i:])
#             # res.extend(lis2[j:])
#             #
#             # return res
#
#         # 这里纯数字列表可以直接互相比较大小
#         # 从第一个元素顺序开始比较，返回第一个不相等元素比较的结果。
#         # 如果所有元素比较均相等，则长的列表大，一样长则两列表相等
#         def bigger(lis1, lis2):
#             for i in range(k):
#                 if lis1[i] < lis2[i]:
#                     return False
#                 elif lis1[i] > lis2[i]:
#                     return True
#             return False
#
#         res = [0] * k
#         for i in range(k + 1):
#             tmp1 = findKmax(nums1, i)
#             tmp2 = findKmax(nums2, k - i)
#             if tmp1 == -1 or tmp2 == -1:
#                 continue
#             tmp = merge(tmp1, tmp2)
#             # if bigger(tmp, res):
#             #     res = tmp
#             if res < tmp:
#                 res = tmp
#         return res
#
#
# solve = Solution()
# print(solve.maxNumber(nums1, nums2, k))

# ############################### 322. 零钱兑换 ################################
# coins = [1, 2, 5]
# amount = 11             # 3
#
# coins = [2]
# amount = 3              # -1
# #
# coins = [1, 2]
# amount = 2              # 1
# #
# coins = [186,419,83,408]
# amount = 6249           # 20
#
# coins = [3,7,405,436]
# amount = 8839
#
#
# class Solution:
#     def coinChange(self, coins, amount) -> int:
#         # coins.sort(reverse=True)
#         # self.res = float("inf")
#         #
#         # def dfs(i, num, amount):
#         #     if amount == 0:
#         #         self.res = min(self.res, num)
#         #         return
#         #     for j in range(i, len(coins)):
#         #         # 剪枝 剩下的最大值都不够凑出来了
#         #         if (self.res - num) * coins[j] < amount:
#         #             break
#         #         if coins[j] > amount:
#         #             continue
#         #         dfs(j, num + 1, amount - coins[j])
#         #
#         # for i in range(len(coins)):
#         #     dfs(i, 0, amount)
#         #
#         # return self.res if self.res != float("inf") else -1
#
#
#         # # 动态规划
#         # dp = [float("inf")] * (amount + 1)
#         # dp[0] = 0
#         # for i in range(1, amount + 1):
#         #     dp[i] = min(dp[i - c] if i - c >= 0 else float("inf") for c in coins) + 1
#         # return dp[-1] if dp[-1] != float("inf") else -1
#
#         dp = [float("inf")] * (amount + 1)
#         dp[0] = 0
#         for i in range(1, amount + 1):
#             for coin in coins:
#                 if i >= coin:
#                     dp[i] = min(dp[i], dp[i - coin] + 1)
#
#         if dp[-1] == float("inf"):
#             return -1
#         return dp[-1]
#
#
# solve = Solution()
# print(solve.coinChange(coins, amount))

# ############################### 324. 摆动排序 II ################################
# # 重新排列成 nums[0] < nums[1] > nums[2] < nums[3]... 的顺序。
# nums = [1, 5, 1, 1, 6, 4]       # [1, 4, 1, 5, 1, 6]
# nums = [1, 3, 2, 2, 3, 1]       # [2, 3, 1, 3, 1, 2]
#
# nums = []       # [2, 3, 1, 3, 1, 2]
#
#
# class Solution:
#     def wiggleSort(self, nums) -> None:
#         """
#         Do not return anything, modify nums in-place instead.
#         """
#         # flag = 0
#         # for i in range(len(nums)):
#         tmp = sorted(nums, reverse=True)
#         cnt = 0
#         length = len(nums) // 2
#         for i in range(len(nums)):
#             if cnt % 2:
#                 nums[i] = tmp[cnt // 2]
#             else:
#                 nums[i] = tmp[cnt // 2 + length]
#             cnt += 1
#
#         import numpy
#         n = len(nums)
#         if n < 2:
#             return
#
#         def A(i):
#             return (2 * i + 1) % (n | 1)
#
#         # find the medium
#         # quick select On
#         key = numpy.median(nums)
#         # i is the start of 2st part
#         # j is the end of 2nd part
#         # k is the end of 3rd part
#         i, j, k = 0, 0, n - 1
#
#         # while True:
#         #     pivot = random.randint(i,k)
#         #     nums[A(pivot)],nums[A(k)] = nums[A(k)],nums[A(pivot)]
#         #     key = nums[A(pivot)]
#         while j <= k:
#             if nums[A(j)] > key:
#                 nums[A(j)], nums[A(i)] = nums[A(i)], nums[A(j)]
#                 i, j = i + 1, j + 1
#             elif nums[A(j)] < key:
#                 nums[A(j)], nums[A(k)] = nums[A(k)], nums[A(j)]
#                 k -= 1
#             else:
#                 j += 1
#
#
# solve = Solution()
# solve.wiggleSort(nums)
# print(nums)

# ############################### 326. 3的幂 ################################
# x = 27      # T
# x = 0       # F
# x = 9       # T
# x = 45      # F
# x = 1       # T
# x = 243
#
#
# class Solution:
#     def isPowerOfThree(self, n: int) -> bool:
#         # if n == 0:
#         #     return False
#         # while n % 3 == 0:
#         #     n //= 3
#         # return n == 1
#
#         # # 3^x = n ==> log3(n) == x 为整数
#         # # 但由于精度原因会出现 4.99999999 的情况: math.log(n, 3) <= 2 * epsilon
#         # if n <= 0:
#         #     return False
#         # import math
#         # return math.log(n, 3) % 1 == 0
#
#         # int能放下的最大3的幂是 3^19 = 1162261467
#         return n > 0 and (1162261467 % n == 0)
#
#
# solve = Solution()
# print(solve.isPowerOfThree(x))

# ############################### 327. 区间和的个数 ################################
# nums = [-2,5,-1]
# lower = -2
# upper = 2           # 3
#
#
# class Solution:
#     def countRangeSum(self, nums, lower, upper):
#
#
# solve = Solution()
# print(solve.countRangeSum(nums, lower, upper))

# ############################### 328. 奇偶链表 ################################


class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


def printList(l: ListNode):
    while l:
        print("%d, " %(l.val), end = '')
        l = l.next
    print('')


def stringToListNode(input):
    # Generate list from the input
    numbers = json.loads(input)

    # Now convert that list into linked list
    dummyRoot = ListNode(0)
    ptr = dummyRoot
    for number in numbers:
        ptr.next = ListNode(number)
        ptr = ptr.next

    ptr = dummyRoot.next
    return ptr


# x = stringToListNode('[1,2,3,4,5]')             # [1,3,5,2,4]
# x = stringToListNode('[1,2,3]')             # [1,3,5,2,4]
# x = stringToListNode('[2,1,3,5,6,4,7]')         # [2,3,6,7,1,5,4]
# x = stringToListNode('[2,1,3,5,6,4]')         # [2,3,6,7,1,5,4]
#
#
# class Solution:
#     def oddEvenList(self, head: ListNode) -> ListNode:
#         # # if not head or not head.next or not head.next.next:
#         # #     return head
#         # odd = ListNode(0)
#         # odd_head = odd
#         # even = ListNode(0)
#         # even_head = even
#         # flag = 0
#         # while head:
#         #     if not flag:
#         #         odd.next = head
#         #         odd = odd.next
#         #         flag = 1
#         #     else:
#         #         even.next = head
#         #         even = even.next
#         #         flag = 0
#         #     head = head.next
#         #
#         # if flag:
#         #     even.next = None
#         # # printList(odd_head.next)
#         # # printList(even_head.next)
#         # odd.next = even_head.next
#         # return odd_head.next
#
#
#         # 不用标志位
#         if not head or not head.next:
#             return head
#         odd = head
#         even = head.next
#         # 偶链表的排头拿出来
#         evenHead = even
#         while even and even.next:
#             odd.next = odd.next.next
#             even.next = even.next.next
#             odd = odd.next
#             even = even.next
#         # 奇链表最后一个和偶数排头连起来
#         odd.next = evenHead
#         return head
#
#
# solve = Solution()
# printList(solve.oddEvenList(x))
#
# ############################### 329. 矩阵中的最长递增路径 ################################
# x = [
#   [9,9,4],
#   [6,6,8],
#   [2,1,1]]          # 4
#
# x = [
#   [3,4,5],
#   [3,2,6],
#   [2,2,1]]          # 4
#
# # x = [[1]]
# # x = [[0,1,2,3,4,5,6,7,8,9],[19,18,17,16,15,14,13,12,11,10],[20,21,22,23,24,25,26,27,28,29],[39,38,37,36,35,34,33,32,31,30],[40,41,42,43,44,45,46,47,48,49],[59,58,57,56,55,54,53,52,51,50],[60,61,62,63,64,65,66,67,68,69],[79,78,77,76,75,74,73,72,71,70],[80,81,82,83,84,85,86,87,88,89],[99,98,97,96,95,94,93,92,91,90],[100,101,102,103,104,105,106,107,108,109],[119,118,117,116,115,114,113,112,111,110],[120,121,122,123,124,125,126,127,128,129],[139,138,137,136,135,134,133,132,131,130],[0,0,0,0,0,0,0,0,0,0]]
# # # 140
#
#
# class Solution:
#     def longestIncreasingPath(self, matrix) -> int:
#         # # 超时
#         # from functools import lru_cache
#         # if not matrix or not matrix[0]:
#         #     return 0
#         # row = len(matrix)
#         # col = len(matrix[0])
#         # res = [1]
#         #
#         # @lru_cache(None)
#         # def backtrack(ii, jj, cnt):
#         #     for x, y in [[-1, 0], [1, 0], [0, 1], [0, -1]]:
#         #         i = x + ii
#         #         j = y + jj
#         #         if 0 <= i < row and 0 <= j < col and matrix[i][j] > matrix[ii][jj]:
#         #             tmp = backtrack(i, j, cnt + 1)
#         #             res[0] = max(res[0], tmp)
#         #     return cnt
#         #
#         # for i in range(row):
#         #     for j in range(col):
#         #         backtrack(i, j, 1)
#         # return res[0]
#
#         # 记忆化回溯
#         if not matrix or not matrix[0]:
#             return 0
#         row = len(matrix)
#         col = len(matrix[0])
#         lookup = [[1] * col for _ in range(row)]
#
#         def backtrack(ii, jj, cnt):
#             if lookup[ii][jj] != 1:
#                 return lookup[ii][jj]
#             res = 1
#             for x, y in [[-1, 0], [1, 0], [0, 1], [0, -1]]:
#                 i = x + ii
#                 j = y + jj
#                 if 0 <= i < row and 0 <= j < col and matrix[i][j] > matrix[ii][jj]:
#                     tmp = backtrack(i, j, cnt + 1)
#                     res = max(res, tmp + 1)
#             lookup[ii][jj] = max(res, lookup[ii][jj])
#             return lookup[ii][jj]
#
#         tmp = 1
#         for i in range(row):
#             for j in range(col):
#                 tmp = max(tmp, backtrack(i, j, 1))
#         return tmp
#
#         # if not matrix or not matrix[0]: return 0
#         #
#         # row = len(matrix)
#         # col = len(matrix[0])
#         # lookup = [[0] * col for _ in range(row)]
#         #
#         # def dfs(i, j):
#         #     if lookup[i][j] != 0:
#         #         return lookup[i][j]
#         #     # 方法一
#         #     res = 1
#         #     for x, y in [[-1, 0], [1, 0], [0, 1], [0, -1]]:
#         #         tmp_i = x + i
#         #         tmp_j = y + j
#         #         if 0 <= tmp_i < row and 0 <= tmp_j < col and \
#         #                 matrix[tmp_i][tmp_j] > matrix[i][j]:
#         #             res = max(res, 1 + dfs(tmp_i, tmp_j))
#         #     lookup[i][j] = max(res, lookup[i][j])
#         #
#         #     # 方法二
#         #     # val = matrix[i][j]
#         #     # lookup[i][j] = 1 + max(
#         #     #     dfs(i + 1, j) if 0 <= i + 1 < row and 0 <= j < col and matrix[i + 1][j] > val else 0,
#         #     #     dfs(i - 1, j) if 0 <= i - 1 < row and 0 <= j < col and matrix[i - 1][j] > val else 0,
#         #     #     dfs(i, j + 1) if 0 <= i < row and 0 <= j + 1 < col and matrix[i][j + 1] > val else 0,
#         #     #     dfs(i, j - 1) if 0 <= i < row and 0 <= j - 1 < col and matrix[i][j - 1] > val else 0,
#         #     # )
#         #     # 方法三
#         #     # lookup[i][j] = 1 + max(
#         #     #     [dfs(i + x, y + j) for x, y in [[-1, 0], [1, 0], [0, 1], [0, -1]] \
#         #     #      if 0 <= (i + x) < row and 0 <= (j + y) < col and matrix[i + x][j + y] > matrix[i][j]] or [0]
#         #     # )
#         #
#         #     return lookup[i][j]
#         #
#         # tmp = 1
#         # for i in range(row):
#         #     for j in range(col):
#         #         tmp = max(tmp, dfs(i, j))
#         # return tmp
#         # # return max(dfs(i, j) for i in range(row) for j in range(col))
#
#
# solve = Solution()
# print(solve.longestIncreasingPath(x))

# ############################### 330. 按要求补齐数组 ################################
nums = [1,3]
n = 6           # 1

# nums = [1,5,10]
# n = 20          # 2

# nums = [1,2,2]
# n = 5           # 0

# nums = [1,2,31,33]
# n = 2147483647


class Solution:
    def minPatches(self, nums, n) -> int:
        # # 利用集合性质,超出内存限制
        # def find_sum(nums, res):
        #     if not nums:
        #         return res
        #     tmp = res.copy()
        #     for ele in res:
        #         tmp.add(ele + nums[0])
        #     return find_sum(nums[1:], res | tmp)
        #
        # rem = find_sum(nums, {0})
        # cnt = 0
        #
        # for i in range(1, n + 1):
        #     if i not in rem:
        #         tmp = set()
        #         for ele in rem:
        #             tmp.add(ele + i)
        #         rem = rem | tmp
        #         cnt += 1
        # return cnt


        count = 0
        miss = 1
        idx = 0
        while miss <= n:
            if idx < len(nums) and nums[idx] <= miss:
                miss += nums[idx]
                idx += 1
            else:
                count += 1
                miss += miss
        return count

        # 若当前可以完全覆盖区间是[1,k]，而当前pos所指向的nums中的元素为B
        # 说明在B之前(因为是升序，所以都比B小)的所有元素之和可以映射到[1, k]
        # 而当我们把B也加入进去后，可映射范围一定向右扩展了B个，也就是变成了[1, k+B]
        # ans = 0
        # current_coverage = 0
        # length = len(nums)
        # pos = 0
        # while current_coverage < n:
        #     if pos < length:
        #         if nums[pos] <= current_coverage + 1:
        #             current_coverage += nums[pos]
        #             pos += 1
        #         else:
        #             ans += 1
        #             current_coverage += current_coverage + 1
        #     else:
        #         ans += 1
        #         current_coverage += current_coverage + 1
        #
        # return ans


solve = Solution()
print(solve.minPatches(nums, n))
