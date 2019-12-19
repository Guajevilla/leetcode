# ############################### 263. 丑数 #################################
# 丑数就是只包含质因数 2, 3, 5 的正整数。
# x = 6       # T
# x = 8       # T
# x = 14      # F
# x = 1      # F
#
#
# class Solution:
#     def isUgly(self, num: int) -> bool:
#         if num <= 0:
#             return False
#         while num % 2 == 0:
#             num //= 2
#         while num % 3 == 0:
#             num //= 3
#         while num % 5 == 0:
#             num //= 5
#         return num == 1
#
#
# solve = Solution()
# print(solve.isUgly(x))

# ############################### 264. 丑数 II #################################
# 找出第 n 个丑数。
# 1, 2, 3, 4, 5, 6, 8, 9, 10, 12 是前 10 个丑数。
n = 10          # 12
n = 58          # 360


class Solution:
    def nthUglyNumber(self, n: int) -> int:
        # 类似动态规划的思路,利用三个指针,从1开始,丑数一定等于2/3/5乘以另一个丑数
        # 将之前的丑数记下来作为乘的因子
        id2 = 0
        id3 = 0
        id5 = 0
        rem = [1]
        for i in range(n - 1):
            rem.append(min(rem[id2]*2, rem[id3]*3, rem[id5]*5))
            if rem[-1] == rem[id2]*2:
                id2 += 1
            if rem[-1] == rem[id3]*3:
                id3 += 1
            if rem[-1] == rem[id5]*5:
                id5 += 1

        return rem[-1]


        # # 最小堆
        # import heapq
        # heap = [1]
        # heapq.heapify(heap)
        # res = 0
        # for _ in range(n):
        #     res = heapq.heappop(heap)
        #     while heap and res == heap[0]:        # res == heap[0]是将重复元素都弹出堆
        #         res = heapq.heappop(heap)
        #     a, b, c = res * 2, res * 3, res * 5
        #     for t in [a, b, c]:
        #         heapq.heappush(heap, t)
        # return res


solve = Solution()
print(solve.nthUglyNumber(n))

# ############################### 268. 缺失数字 #################################
# # 你的算法应具有线性时间复杂度。你能否仅使用额外常数空间来实现?
# x = [3,0,1]                 # 2
# # x = [9,6,4,2,3,5,7,0,1]     # 8
#
#
# class Solution:
#     def missingNumber(self, nums):
#         # # 求和公式
#         # n = len(nums)
#         # return (n + 1) * n // 2 - sum(nums)
#         # # 但是求和公式有溢出的风险,以下是改进,边加边减
#         # res = len(nums)
#         # for idx, num in enumerate(nums):
#         #     res += idx - num
#         # return res
#
#
#         # 位运算
#         # 先得到 [0..n] 的异或值，再将结果对数组中的每一个数进行一次异或运算。
#         # 未缺失的数在 [0..n] 和数组中各出现一次，因此异或后得到 0。
#         # 而缺失的数字只在 [0..n] 中出现了一次，在数组中没有出现，因此最终的异或结果即为这个缺失的数字。
#         missing = len(nums)
#         for i, num in enumerate(nums):
#             missing ^= i ^ num
#         return missing
#
#
#         # 排序法
#         # nums.sort()
#         # left = 0
#         # right = len(nums)
#         # while left < right:
#         #     mid = left + (right - left) // 2
#         #     if nums[mid] > mid:
#         #         right = mid
#         #     else:
#         #         left = mid + 1
#         # return left
#
#
# solve = Solution()
# print(solve.missingNumber(x))
