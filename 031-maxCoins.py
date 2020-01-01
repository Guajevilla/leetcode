# ############################### 312. 戳气球 ################################
x = [3,1,5,8]           # 167   coins = 3*1*5 + 3*5*8 + 1*3*8 + 1*8*1 = 167
# x = [35,16,83,87,84,59,48,41,20,54]
# x = [3,1,5]             # 35
# x = [9]             # 35


class Solution:
    def maxCoins(self, nums) -> int:
        # 超时而且好像有bug
        # res = [0]
        #
        # def backtrack(nums, tmp):
        #     if len(nums) == 1:
        #         return nums[0] + tmp
        #     for i in range(len(nums)):
        #         if i == 0:
        #             ttmp = nums[0] * nums[1]
        #         elif i == len(nums) - 1:
        #             ttmp = nums[-2] * nums[-1]
        #         else:
        #             ttmp = nums[i-1] * nums[i] * nums[i+1]
        #         res[0] = max(res[0], backtrack(nums[:i]+nums[i+1:], tmp + ttmp))
        #     return res[0]
        #
        # # backtrack(nums, 0)
        # return backtrack(nums, 0)

        '''
        大区间能分解出独立的小区间，我们可以先算小区间，并组合出大区间。
        i是起始 j是终点 k是宽度
        从2开始，层层迭代
        t是i+1~j-1
        '''
        nums = [1] + nums + [1]
        n = len(nums)
        dp = [[0] * n for _ in range(n)]
        for k in range(2, n):
            for i in range(n - k):
                j = i + k
                for t in range(i + 1, j):
                    dp[i][j] = max(dp[i][j], nums[i] * nums[t] * nums[j] + dp[i][t] + dp[t][j])
        return dp[0][n - 1]


solve = Solution()
print(solve.maxCoins(x))

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

# ############################# 315. 计算右侧小于当前元素的个数 ##############################
# x = [5,2,6,1]           # [2,1,1,0]
# x = [2,0,1]             # [2,0,0]
#
#
# class Solution:
#     def countSmaller(self, nums):
#         dp = [0] * len(nums)
#         for i in range(len(nums)-2, -1, -1):
#             j = i + 1
#             while j < len(nums):
#                 if nums[j] < nums[i]:
#                     dp[i] = dp[j] + 1
#                     break
#                 elif nums[j] == nums[i]:
#                     dp[i] = dp[j]
#                     break
#                 else:
#                     j += 1
#         return dp
#
#         # # 利用折半排序
#         # import bisect
#         # queue = []
#         # res = []
#         # for num in nums[::-1]:
#         #     loc = bisect.bisect_left(queue, num)
#         #     res.append(loc)
#         #     queue.insert(loc, num)
#         # return res[::-1]
#
#
# solve = Solution()
# print(solve.countSmaller(x))

# ############################# 316. 去除重复字母 ##############################
# # 给定一个仅包含小写字母的字符串，去除字符串中重复的字母，使得每个字母只出现一次。
# # 需保证返回结果的字典序最小（要求不能打乱其他字符的相对位置）。
# x = "bcabc"         # "abc"
# x = "cbacdcbc"      # "acdb"
# # x = "cdbeacdcbc"    # "beacd"
#
#
# class Solution:
#     def removeDuplicateLetters(self, s: str) -> str:
#         # stack = []
#         # for i, ele in enumerate(s):
#         #     if not stack:
#         #         stack.append(ele)
#         #     elif ord(ele) <= ord(stack[-1]):
#         #         if ele not in stack:
#         #             # for j in stack:
#         #             #     if ord(j) > ord(ele) and j in s[i+1:]:
#         #             #         stack.remove(j)
#         #             j = i - 1
#         #             while j >= 0:
#         #                 if stack and ord(stack[-1]) > ord(ele) and stack[-1] in s[i + 1:]:
#         #                     stack.pop(-1)
#         #                     j -= 1
#         #                 else:
#         #                     break
#         #
#         #             stack.append(ele)
#         #     else:
#         #         # if ele in stack:
#         #         #     stack.remove(ele)
#         #         # stack.append(ele)
#         #         if ele not in stack:
#         #             stack.append(ele)
#         # return ''.join(stack)
#
#
#         # size = len(s)
#         #
#         # last_appear_index = [0 for _ in range(26)]
#         # if_in_stack = [False for _ in range(26)]
#         #
#         # # 记录每个字符最后一次出现的索引
#         # for i in range(size):
#         #     last_appear_index[ord(s[i]) - ord('a')] = i
#         #
#         # stack = []
#         # for i in range(size):
#         #     if if_in_stack[ord(s[i]) - ord('a')]:
#         #         continue
#         #
#         #     while stack and ord(stack[-1]) > ord(s[i]) and \
#         #             last_appear_index[ord(stack[-1]) - ord('a')] >= i:
#         #         top = stack.pop()
#         #         if_in_stack[ord(top) - ord('a')] = False
#         #
#         #     stack.append(s[i])
#         #     if_in_stack[ord(s[i]) - ord('a')] = True
#         #
#         # return ''.join(stack)
#
#         # counts = {}
#         # for c in s:  # 字符计数
#         #     counts[c] = counts.get(c, 0) + 1
#         #
#         # stack, stacked = ['0'], set()   # stack为答案，放置哨兵，stacked为stack中已有的字符
#         # for c in s:
#         #     if c not in stacked:
#         #         while c < stack[-1] and counts[stack[-1]]:  # 当栈顶在后面还有且大于当前字符时弹出
#         #             stacked.remove(stack.pop())
#         #         stack.append(c)
#         #         stacked.add(c)
#         #     counts[c] -= 1
#         # return ''.join(stack[1:])
#
#
#         # 递归,逐个找字典最小的字母
#         # 先按字典排序
#         for a in sorted(set(s)):
#             tmp = s[s.index(a):]
#             # 看余下的是否能组成所需的字母
#             if len(set(tmp)) == len(set(s)):
#                 return a + self.removeDuplicateLetters(tmp.replace(a, ""))
#         return ""
#
#         # 相同思想,迭代实现
#         res = ""
#         while s:
#             # 从右往左找，找到最小位置的索引号
#             loc = min(map(s.rindex, s))
#             # 找该索引前面最小的字母
#             a = min(s[:loc + 1])
#             res += a
#             s = s[s.index(a):].replace(a, "")
#         return res
#
#
# solve = Solution()
# print(solve.removeDuplicateLetters(x))

# ############################# 318. 最大单词长度乘积 ##############################
# x = ["abcw", "baz", "foo", "bar", "xtfn", "abcdef"]  # 16
# x = ["a", "ab", "abc", "d", "cd", "bcd", "abcd"]  # 4
# x = ["a", "aa", "aaa", "aaaa"]  # 0
#
#
# class Solution:
#     def maxProduct(self, words) -> int:
#         # 利用集合的交 (x & y) 并 (x | y) 差 (x - y) 集操作
#         rem = []
#         rem_len = []
#         for word in words:
#             rem.append(set(word))
#             rem_len.append(len(word))
#         # print(rem)
#         res = 0
#         for i, s1 in enumerate(rem):
#             for j in range(i+1, len(rem)):
#                 if not s1 & rem[j]:
#                     res = max(res, rem_len[i] * rem_len[j])
#         return res
#
#
#         # # 用位运算代替单词,用26位表示一个单词
#         # values = [0] * len(words)
#         # # 用位运算表示一个单词
#         # for i in range(len(words)):
#         #     for alp in words[i]:
#         #         values[i] |= 1 << (ord(alp) - 97)
#         # return max([len(words[i]) * len(words[j]) for i in range(len(words)) for j in range(i, len(words)) if
#         #            not values[i] & values[j]] or [0])
#         #
#         #
#         # # 位运算加哈希
#         # from collections import defaultdict
#         # lookup = defaultdict(int)
#         # for i in range(len(words)):
#         #     mask = 0
#         #     for alp in words[i]:
#         #         mask |= 1 << (ord(alp) - 97)
#         #     lookup[mask] = max(lookup[mask], len(words[i]))
#         # #print(lookup)
#         # return max([lookup[x] * lookup[y] for x in lookup for y in lookup if not x & y] or [0])
#
#
# solve = Solution()
# print(solve.maxProduct(x))

# ############################# 319. 灯泡开关 ##############################
# 第i个灯泡的反转次数等于它所有因子（包括1和i）的个数
# 一开始的状态的灭的，只有反转奇数次才会变成亮的，所以只有因子个数为奇数的灯泡序号才会亮
# 只有平方数的因子数为奇数（比如6=1*6,2*3，它们的因子总是成对出现的，而4=1*4,2*2，只有平方数的平方根因子会只出现1次）
# 所以最终答案等于n以内（包括n和1）的平方数数量，只要计算sqrt(n)即可
# import math
# x = 3       # 1
#
#
# class Solution:
#     def bulbSwitch(self, n: int) -> int:
#         # 暴力法超时
#         # dp = [0] * n
#         # for i in range(n):
#         #     for j in range(i, n, i+1):
#         #         dp[j] = 1 - dp[j]
#         #     print(dp)
#         # return sum(dp)
#
#         # 可以多打印几个结果找规律
#         return int(math.sqrt(n))
#
#
# solve = Solution()
# for i in range(1, 50):
#     print(solve.bulbSwitch(i))
