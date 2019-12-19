import math
# ############################### 273. 整数转换英文表示 #################################
x = 123             # "One Hundred Twenty Three"
x = 12345           # "Twelve Thousand Three Hundred Forty Five"
x = 1234567         # "One Million Two Hundred Thirty Four Thousand Five Hundred Sixty Seven"
x = 1234567891      # "One Billion Two Hundred Thirty Four Million Five Hundred Sixty Seven Thousand Eight Hundred Ninety One"
x = 1000000


class Solution:
    def numberToWords(self, num: int) -> str:
        # if num == 0:
        #     return 'Zero'
        # words1 = ['One','Two','Three','Four','Five','Six','Seven','Eight','Nine','Ten','Eleven','Twelve','Thirteen','Fourteen','Fifteen','Sixteen','Seventeen','Eighteen','Nineteen']
        # words2 = ['Ten','Twenty','Thirty','Forty','Fifty','Sixty','Seventy','Eighty','Ninety']
        #
        # def wordOf3Num(num):
        #     res = ''
        #     if num // 100:
        #         tmp = num // 100
        #         num = num % 100
        #         res += words1[tmp - 1] + " Hundred "
        #     if num >= 20:
        #         tmp = num // 10
        #         num = num % 10
        #         res += words2[tmp - 1]
        #         if num:
        #             res += " " + words1[num - 1]
        #     elif num > 0:
        #         res += words1[num - 1]
        #     return res.strip()
        #
        # res = ''
        # bi = num // (10 ** 9)
        # num %= (10 ** 9)
        # if bi != 0:
        #     res += words1[bi - 1] + ' Billion '
        # mi = num // (10 ** 6)
        # num %= (10 ** 6)
        # if mi != 0:
        #     res += wordOf3Num(mi) + ' Million '
        # th = num // (10 ** 3)
        # num %= (10 ** 3)
        # if th != 0:
        #     res += wordOf3Num(th) + ' Thousand '
        # res += wordOf3Num(num)
        # return res.strip()

        # 这个写的更简洁
        to19 = 'One Two Three Four Five Six Seven Eight Nine Ten Eleven Twelve ' \
               'Thirteen Fourteen Fifteen Sixteen Seventeen Eighteen Nineteen'.split()
        tens = 'Twenty Thirty Forty Fifty Sixty Seventy Eighty Ninety'.split()

        def helper(num):
            if num < 20:
                return to19[num - 1:num]
            if num < 100:
                return [tens[num // 10 - 2]] + helper(num % 10)
            if num < 1000:
                return [to19[num // 100 - 1]] + ["Hundred"] + helper(num % 100)
            for p, w in enumerate(["Thousand", "Million", "Billion"], 1):
                if num < 1000 ** (p + 1):
                    return helper(num // 1000 ** p) + [w] + helper(num % 1000 ** p)

        return " ".join(helper(num)) or "Zero"


solve = Solution()
print(solve.numberToWords(x))

# ############################### 274. H指数 #################################
# # 如果 h 有多种可能的值，h 指数是其中最大的那个。
# x = [3,0,6,1,5]         # 3
# # x = [2,0,6,1,5]         # 2
# # x = [1, 2]              # 1
#
#
# class Solution:
#     def hIndex(self, citations):
#         # n = len(citations)
#         # citations.sort(reverse=True)
#         # for i, num in enumerate(citations):
#         #     if num <= i:
#         #         return i
#         # return n
#
#
#         # 桶排序
#         # 因为 H指数 一定小于等于论文的数量n，所以把引用量大于论文数量的放在一起
#         # 分成 n+1 个桶
#         n = len(citations)
#         bucket = [0] * (n + 1)
#         for citation in citations:
#             if citation >= n:
#                 bucket[n] += 1
#             else:
#                 bucket[citation] += 1
#         print(bucket)
#         cur = 0
#         for i in range(n, -1, -1):
#             cur += bucket[i]
#             if cur >= i:
#                 return i
#
#
# solve = Solution()
# print(solve.hIndex(x))

# ############################### 275. H指数 II #################################
# 数组已经按照升序排列, 优化你的算法到对数时间复杂度
# x = [0,1,3,5,6]         # 3
# x = [5]         # 3
#
#
# class Solution:
#     def hIndex(self, citations):
#         # for i, num in enumerate(citations[::-1]):
#         #     if num <= i:
#         #         return i
#         # return len(citations)
#
#         # 二分法
#         n = len(citations)
#         l = 0
#         r = n - 1
#         res = 0
#         while l <= r:
#             mid = l + (r - l) // 2
#             if citations[mid] >= n - mid:
#                 res = n - mid
#                 r = mid - 1
#             else:
#                 l = mid + 1
#         return res
#
#
# solve = Solution()
# print(solve.hIndex(x))

# ############################### 278. 第一个错误的版本 #################################
# n = 5
# k = 4
#
#
# def isBadVersion(version):
#     return version >= k
#
#
# class Solution:
#     def firstBadVersion(self, n):
#         # 二分法要注意左右两个指针移动的时候取值与mid的关系
#         """
#         :type n: int
#         :rtype: int
#         """
#         l = 1
#         r = n
#         while l < r:
#             # mid = (r + l) // 2
#             # 这个地方讲道理应该写成:
#             # 因为直接加可能会溢出
#             mid = l + (r - l) // 2
#             if isBadVersion(mid):
#                 r = mid
#             else:
#                 l = mid + 1
#         return r
#
#
# solve = Solution()
# print(solve.firstBadVersion(x))

# ############################### 279. 完全平方数 #################################
# n = 13      # 2
# n = 12      # 3
# n = 7       # 4
# # n = 5374
#
#
# class Solution:
#     # def numSquares(self, n: int) -> int:
#     #     # # 动态规划
#     #     # # 这题我把 j**2 改写成 j*j就过了..也不知道为什么..
#     #     # dp = [0] * (n + 1)
#     #     # for i in range(1, n + 1):
#     #     #     dp[i] = dp[i - 1]
#     #     #     j = 2
#     #     #     while j*j <= i:
#     #     #         dp[i] = min(dp[i-j*j], dp[i])
#     #     #         j += 1
#     #     #     dp[i] += 1
#     #     #
#     #     # return dp[-1]
#     #     #
#     #     # # dp = [i for i in range(n + 1)]
#     #     # # for i in range(2, n + 1):
#     #     # #     for j in range(1, int(i ** (0.5)) + 1):
#     #     # #         dp[i] = min(dp[i], dp[i - j * j] + 1)
#     #     # # return dp[-1]
#     #
#     #     # BFS,其实感觉与动态规划的意思类似
#     #     from collections import deque
#     #     if n == 0: return 0
#     #     queue = deque([n])
#     #     step = 0
#     #     visited = set()
#     #     while queue:
#     #         step += 1
#     #         l = len(queue)
#     #         for _ in range(l):
#     #             tmp = queue.pop()
#     #             for i in range(1, int(tmp ** 0.5) + 1):
#     #                 x = tmp - i ** 2
#     #                 if x == 0:
#     #                     return step
#     #                 if x not in visited:
#     #                     queue.appendleft(x)
#     #                     visited.add(x)
#     #     return step
#
#     # 数学方法 四平方和定理
#     # 任何正整数都可以拆分成不超过4个数的平方和 --> 答案只可能是1, 2, 3, 4
#     # 如果一个数最少可以拆成4个数的平方和，则这个数还满足
#     # n = (4 ^ a) * (8b + 7)
#     # 如果这个数本来就是某个数的平方，那么答案就是1
#     # 如果答案是2，即n = a ^ 2 + b ^ 2，那么我们可以枚举a，来验证
#     # 如果验证通过则答案是2否则只能是3
#     def isSquare(self, n: int) -> bool:
#         sq = int(math.sqrt(n))
#         return sq * sq == n
#
#     def numSquares(self, n: int) -> int:
#         # Lagrange's four-square theorem
#         if self.isSquare(n):
#             return 1
#         while (n & 3) == 0:     # 对4取余等于0 (即能被4整除)
#             n >>= 2
#         if (n & 7) == 7:        # 对8取余等于7
#             return 4
#         sq = int(math.sqrt(n)) + 1
#         for i in range(1, sq):
#             if self.isSquare(n - i * i):
#                 return 2
#         return 3
#
#
# solve = Solution()
# print(solve.numSquares(n))
