# ############################### 312. 戳气球 ################################
x = [3,1,5,8]           # 167


# class Solution:
#     def maxCoins(self, nums) -> int:
#
#
# solve = Solution()
# print(solve.maxCoins(x))

# ############################### 313. 超级丑数 ################################
# n = 12
# primes = [2,7,13,19]
#
#
# class Solution:
#     def nthSuperUglyNumber(self, n: int, primes) -> int:
#         # idx = [0] * len(primes)
#         # rem = [1]
#         # for i in range(n - 1):
#         #     tmp = float("inf")
#         #     for j in range(len(primes)):
#         #         tmp = min(tmp, rem[idx[j]]*primes[j])
#         #     rem.append(tmp)
#         #
#         #     for j in range(len(primes)):
#         #         if rem[-1] == rem[idx[j]] * primes[j]:
#         #             idx[j] += 1
#         #
#         # return rem[-1]
#
#
#         # 利用堆,对每次的质数表乘以当前最小丑数,并把这些数压入堆中,每次出堆最小值
#         import heapq
#         heap = [1]
#         n -= 1
#         while n:
#             tmp = heapq.heappop(heap)
#             while heap and tmp == heap[0]:
#                 tmp = heapq.heappop(heap)
#             for p in primes:
#                 t = p * tmp
#                 heapq.heappush(heap, t)
#             n -= 1
#         return heapq.heappop(heap)
#
#
# solve = Solution()
# print(solve.nthSuperUglyNumber(n, primes))
