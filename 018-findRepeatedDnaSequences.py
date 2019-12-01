import math
import json
# ############################### 187. 重复的DNA序列 ###############################
# # 查找 DNA 分子中所有出现超过一次的 10 个字母长的序列
# s = "AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT"      # ["AAAAACCCCC", "CCCCCAAAAA"]
# s = "AAAAACCCCCAAAAACCCCCCAAAAAGGGTTTAAAAACCCCC"      # ["AAAAACCCCC", "CCCCCAAAAA"]
#
#
# class Solution:
#     def findRepeatedDnaSequences(self, s: str):
#         # res = set()
#         # rem = set()
#         # for i in range(len(s) - 9):
#         #     tmp = s[i: i + 10]
#         #     if tmp in rem:
#         #         res.add(tmp)
#         #     rem.add(tmp)
#         # return list(res)
#
#         # 同样的思路,上面用两个set一个用于记录,一个用于防止答案重复出现
#         # 下面这个用一个普通列表和一个字典,字典存出现次数来防止重复
#         res = []
#         rem = {}
#         for i in range(len(s) - 9):
#             tmp = s[i: i + 10]
#             if tmp in rem:
#                 if rem[tmp] == 1:
#                     res.append(tmp)
#                 rem[tmp] += 1
#             else:
#                 rem[tmp] = 1
#         return list(res)
#
#
# solve = Solution()
# print(solve.findRepeatedDnaSequences(s))

# ############################### 188. 买卖股票的最佳时机 IV ###############################
# class Solution:
#     def maxProfit(self, k: int, prices: List[int]) -> int:

# ############################### 189. 旋转数组 ###############################
# # 尽可能想出更多的解决方案，至少有三种不同的方法可以解决这个问题。
# # 要求使用空间复杂度为 O(1) 的 原地 算法。
# nums = [1,2,3,4,5,6,7]
# k = 3                   # [5,6,7,1,2,3,4]
#
# nums = [-1,-100,3,99]
# k = 2                   # [3,99,-1,-100]
#
#
# class Solution:
#     def rotate(self, nums, k):
#         """
#         Do not return anything, modify nums in-place instead.
#         """
#         # # for i in range(k):
#         # #     nums.insert(0, nums.pop())
#         #
#         # # 三次反转
#         # [4,3,2,1,5,6,7] ->
#         # [4,3,2,1,7,6,5] ->
#         # [5,6,7,1,2,3,4]
#         # n = len(nums)
#         # k = k % n
#         #
#         # def swap(l, r):
#         #     while (l < r):
#         #         nums[l], nums[r] = nums[r], nums[l]
#         #         l = l + 1
#         #         r = r - 1
#         #
#         # swap(0, n - k - 1)
#         # swap(n - k, n - 1)
#         # swap(0, n - 1)
#
#         # 利用切片
#         n = len(nums)
#         k %= n
#         nums[:] = nums[n - k:] + nums[:n - k]
#
#
# solve = Solution()
# solve.rotate(nums, k)
# print(nums)

# ############################### 190. 颠倒二进制位 ###############################
x = int('00000010100101000001111010011100', 2)
ans = int('00111001011110000010100101000000', 2)

x = int('11111111111111111111111111111101', 2)
ans = int('10111111111111111111111111111111', 2)


class Solution:
    def reverseBits(self, n: int) -> int:
        # # [2:]去掉前面的二进制标志0b,zfill(32)左补全为32位
        # s = str(bin(n))[2:].zfill(32)
        # res = int(s[::-1], 2)
        # return res

        # 一位一位比
        res = 0
        count = 32

        while count:
            res <<= 1
            # 取出 n 的最低位数加到 res 中
            res += n & 1
            n >>= 1
            count -= 1

        return int(bin(res), 2)


solve = Solution()
print(solve.reverseBits(x) == ans)
