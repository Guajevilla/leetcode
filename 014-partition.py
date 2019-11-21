# ############################### 131. 分割回文串 ###############################
# nums = "aab"
# # [
# #   ["aa","b"],
# #   ["a","a","b"]
# # ]
#
# nums = "abbabbc"
# # [
# #   ["abba","bb","c"],
# #   ["abba","b","b","c"],
# #   ["a","bb","a","bb","c"],
# #   ["a","b","b","a","bb","c"],
# #   ["a","bb","a","b","b","c"],
# #   ["a","b","b","a","b","b","c"],
# #   ["a","bbabb","c"],
# #   ["a","b","bab","b","c"],
# # ]
#
# # nums = ""
#
#
# class Solution(object):
#     def partition(self, s):
#         n = len(s)
#         dp = [[False] * n for _ in range(n)]
#
#         # dp是i到j判断回文串
#         for i in range(n):
#             for j in range(i + 1):
#                 # 尽量利用之前的信息判断回文串,i - j <= 2是如果头尾相等且全长小于等于3,dp[j + 1][i - 1]是去掉头尾是否是回文串
#                 if (s[i] == s[j]) and (i - j <= 2 or dp[j + 1][i - 1]):
#                     dp[j][i] = True
#         # print(dp)
#         res = []
#
#         def helper(i, tmp):
#             if i == n:
#                 res.append(tmp)
#             for j in range(i, n):
#                 if dp[i][j]:
#                     helper(j + 1, tmp + [s[i: j + 1]])
#
#         helper(0, [])
#         return res
#
#
#     # # 直接用s[:i] == s[:i][::-1]判断回文,不用循环
#     # # def is_palindrome(self, s):
#     # #     i = 0
#     # #     j = len(s) - 1
#     # #     while i < j:
#     # #         if s[i] != s[j]:
#     # #             return False
#     # #         i += 1
#     # #         j -= 1
#     # #     return True
#     #
#     # def partition(self, s):
#     #     """
#     #     :type s: str
#     #     :rtype: List[List[str]]
#     #     """
#     #     res = []
#     #
#     #     def sub_solver(s, tmp):
#     #         if not s:
#     #             res.append(tmp)
#     #             return
#     #         for i in range(1, len(s) + 1):
#     #             # if self.is_palindrome(s[:i]):
#     #             if s[:i] == s[:i][::-1]:
#     #                 sub_solver(s[i:], tmp + [s[:i]])
#     #
#     #     sub_solver(s, [])
#     #     return res
#
#
# solve = Solution()
# print(solve.partition(nums))

# ############################### 132. 分割回文串 II ###############################
# nums = "aab"
# # 1
#
# nums = "abbabbc"
# # 2
#
# # nums = ""
# # # 0
#
#
# class Solution(object):
#     def minCut(self, s):
#         """
#         :type s: str
#         :rtype: int
#         """
#         # dp是i到j判断回文串
#         min_s = list(range(len(s)))
#         n = len(s)
#         dp = [[False] * n for _ in range(n)]
#         for i in range(n):
#             for j in range(i + 1):
#                 if s[i] == s[j] and (i - j < 2 or dp[j + 1][i - 1]):
#                     dp[j][i] = True
#                     # 说明不用分割
#                     if j == 0:
#                         min_s[i] = 0
#                     else:
#                         min_s[i] = min(min_s[i], min_s[j - 1] + 1)
#         return min_s[-1]
#
#         # 暴力判断
#         # dp = [len(s) for i in range(len(s) + 1)]
#         #
#         # dp[0] = -1
#         # for i in range(len(s)):
#         #     for j in range(i + 1):
#         #         if s[j:i + 1] == s[j:i + 1][::-1]:
#         #             dp[i + 1] = min(dp[i + 1], dp[j] + 1)
#         #
#         # return dp[-1]
#
#
# solve = Solution()
# print(solve.minCut(nums))

# ############################### 133. 克隆图 ###############################
#
#
# # Definition for a Node.
# class Node(object):
#     def __init__(self, val, neighbors):
#         self.val = val
#         self.neighbors = neighbors
#
#
# class Solution(object):
#     def cloneGraph(self, node):
#         """
#         :type node: Node
#         :rtype: Node
#         """
#         lookup = {}
#
#         def dfs(node):
#             # print(node.val)
#             if not node: return
#             if node in lookup:
#                 return lookup[node]
#             clone = Node(node.val, [])
#             lookup[node] = clone
#             for n in node.neighbors:
#                 clone.neighbors.append(dfs(n))
#
#             return clone
#
#         return dfs(node)
#
#
#         # head = Node(node.val, [])
#         # rem = {}
#         #
#         # def solver(node, res):
#         #     # if res.neighbors:
#         #     #     return
#         #     # res.val = node.val
#         #     rem[node] = res
#         #     for neighbor in node.neighbors:
#         #         if neighbor not in rem:
#         #             res.neighbors.append(Node(neighbor.val, []))
#         #             solver(neighbor, res.neighbors[-1])
#         #         else:
#         #             res.neighbors.append(rem[neighbor])
#         #
#         # solver(node, head)
#         # return head
#
#
# solve = Solution()
# print(solve.cloneGraph(nums))

# ############################### 134. 加油站 ###############################
# gas  = [1,2,3,4,5]
# cost = [3,4,5,1,2]      # 3
#
# # gas  = [2,3,4]
# # cost = [3,4,3]          # -1
#
#
# class Solution(object):
#     def canCompleteCircuit(self, gas, cost):
#         """
#         :type gas: List[int]
#         :type cost: List[int]
#         :rtype: int
#         """
#         # 只要gas和大于cost和,就一定有解
#         if sum(gas) < sum(cost): return -1
#         res = 0
#         tank = 0
#         # 如果第i个加油站不满足条件,那么必不可能从i号加油站及其之前的加油站出发
#         for i, item in enumerate(zip(gas, cost)):
#             tank += (item[0] - item[1])
#             if tank < 0:
#                 res = i + 1
#                 tank = 0
#         return res
#
#
# solve = Solution()
# print(solve.canCompleteCircuit(gas, cost))

# ############################### 135. 分发糖果 ###############################
# # 每个孩子至少分配到 1 个糖果.
# # 相邻的孩子中,评分高的孩子必须获得更多的糖果.
# x = [1,0,2]     # 5 [2,1,2]
# # x = [1,2,2]     # 4 [1,2,1]
# # x = [1,2,3]     # 6 [1,2,3]
# x = [1,2,3,2,1,4]
#
#
# class Solution(object):
#     def candy(self, ratings):
#         """
#         :type ratings: List[int]
#         :rtype: int
#         """
#         # n = len(ratings)
#         # if n == 0:
#         #     return 0
#         # candy_nums = [1] * n
#         #
#         # for i in range(1, n):
#         #     if ratings[i] > ratings[i - 1]:
#         #         candy_nums[i] = candy_nums[i - 1] + 1
#         #
#         # for i in range(n - 1, 0, -1):
#         #     if ratings[i - 1] > ratings[i]:
#         #         candy_nums[i - 1] = max(candy_nums[i - 1], candy_nums[i] + 1)
#         # # print(candy_nums)
#         # return sum(candy_nums)
#
#         # 只扫一遍
#         res = 1
#         # 先前值
#         pre = 1
#         # 递减长度
#         des_num = 0
#         for i in range(1, len(ratings)):
#             if ratings[i] >= ratings[i - 1]:
#                 if des_num > 0:
#                     # 求和公式
#                     res += ((1 + des_num) * des_num) // 2
#                     # 递减长度比先前值大,所以我们要把先前值补充
#                     if pre <= des_num: res += (des_num - pre + 1)
#                     pre = 1
#                     des_num = 0
#                 if ratings[i] == ratings[i - 1]:
#                     pre = 1
#                 else:
#                     pre += 1
#                 res += pre
#             else:
#                 des_num += 1
#         # print(des_num)
#         if des_num > 0:
#             res += ((1 + des_num) * des_num) // 2
#             if pre <= des_num: res += (des_num - pre + 1)
#         return res
#
#
# solve = Solution()
# print(solve.candy(x))

# ############################### 136. 只出现一次的数字 ###############################
# x = [2,2,1]     # 1
# # x = [4,1,2,1,2]     # 4
#
#
# class Solution(object):
#     def singleNumber(self, nums):
#         """
#         :type nums: List[int]
#         :rtype: int
#         """
#         # rem = []
#         # for num in nums:
#         #     if num not in rem:
#         #         rem.append(num)
#         #     else:
#         #         rem.remove(num)
#         # return rem[0]
#
#         # 利用集合无重复性
#         return sum(set(nums)) * 2 - sum(nums)
#
#         # 利用异或,相同的数异或为0,
#         # 且有交换律 a ^ b ^ c = a ^ c ^ b
#         # 也可理解为二进制下不考虑进位的加法
#         res = nums[0]
#         for num in nums[1:]:
#             res ^= num
#         return res
#
#
# solve = Solution()
# print(solve.singleNumber(x))

# ############################### 137. 只出现一次的数字 II ###############################
# x = [2,2,3,2]       # 3
# x = [0,1,0,1,0,1,99]       # 99
#
#
# class Solution(object):
#     def singleNumber(self, nums):
#         """
#         :type nums: List[int]
#         :rtype: int
#         """
#         # 如果能设计一个状态转换电路，使得一个数出现3次时能自动抵消为0，最后剩下的就是只出现1次的数。
#         # 相当于模拟三进制下不考虑进位的加法(使得 0 # 1 = 1; 1 # 1 = 2; 2 # 1 = 0)
#         # 因为保存状态有三个所以需要两位
#         # x & ~x = 0;
#         # x & ~0 = x
#         ones, twos = 0, 0
#         for num in nums:
#             ones = ones ^ num & ~twos   # & ~twos用来判断是不是第三次出现
#             twos = twos ^ num & ~ones
#         return ones
#
#
#         # # 利用集合,但set需要用到O(n)额外空间,而且还有数越界的可能
#         # return (sum(set(nums)) * 3 - sum(nums)) // 2
#
#
# solve = Solution()
# print(solve.singleNumber(x))

# ############################### 138. 复制带随机指针的链表 ###############################
#
#
# # Definition for a Node.
# class Node(object):
#     def __init__(self, val, next, random):
#         self.val = val
#         self.next = next
#         self.random = random
#
#
# x = Node(-1, None, None)
#
#
# class Solution(object):
#     def copyRandomList(self, head):
#         """
#         :type head: Node
#         :rtype: Node
#         """
#         # 第一遍先复制链表,并记录复制与原节点的对应关系,再第二遍复制random
#         if not head:
#             return head
#         dummy = Node(0, head, None)
#         p = dummy
#         dic = {None: None}
#         while p.next:
#             tmp = Node(p.next.val, p.next.next, None)
#             dic[p.next] = tmp
#             p.next = tmp
#             p = p.next
#
#         p = dummy.next
#         while p:
#             p.random = dic[head.random]
#             p = p.next
#             head = head.next
#
#         return dummy.next
#
#         # # DFS,做法同133
#         # lookup = {}
#         #
#         # def dfs(head):
#         #     if not head: return None
#         #     if head in lookup: return lookup[head]
#         #     clone = Node(head.val, None, None)
#         #     lookup[head] = clone
#         #     clone.next, clone.random = dfs(head.next), dfs(head.random)
#         #     return clone
#         #
#         # return dfs(head)
#
#         # 倍增法,优点在于不需要额外空间
#         # 在每个节点后紧跟着复制原节点,再复制random,最后断开连接,变为两个链表
#         if not head: return None
#         # 复制节点
#         cur = head
#         while cur:
#             # 保存下一个节点
#             tmp = cur.next
#             # 后面跟着同样的节点
#             cur.next = Node(cur.val, None, None)
#             # 拼接
#             cur.next.next = tmp
#             cur = tmp
#         # 复制random指针
#         cur = head
#         while cur:
#             if cur.random:
#                 cur.next.random = cur.random.next
#             cur = cur.next.next
#         # 拆分
#         cur = head
#         copy_head = head.next
#         copy_cur = copy_head
#         while copy_cur.next:
#             # 组head
#             cur.next = cur.next.next
#             cur = cur.next
#             # 组 copy_head
#             copy_cur.next = copy_cur.next.next
#             copy_cur = copy_cur.next
#         # 把链表结束置空
#         cur.next = copy_cur.next
#         copy_cur.next = None
#         return copy_head
#
#
# solve = Solution()
# print(solve.copyRandomList(x))

# ############################### 139. 单词拆分 ###############################
s = "leetcode"              # T
wordDict = ["leet", "code"]

# s = "applepenapple"         # T
# wordDict = ["apple", "pen"]

# s = "catsandog"             # F
# wordDict = ["cats", "dog", "sand", "and", "cat"]

# s = "catsandog"             # T
# wordDict = ["cat", "og", "and", "cats"]

s = "acaaaaabbbdbcccdcdaadcdccacbcccabbbbcdaaaaaadb"
wordDict = ["abbcbda","cbdaaa","b","dadaaad","dccbbbc","dccadd","ccbdbc","bbca","bacbcdd","a","bacb","cbc","adc","c","cbdbcad","cdbab","db","abbcdbd","bcb","bbdab","aa","bcadb","bacbcb","ca","dbdabdb","ccd","acbb","bdc","acbccd","d","cccdcda","dcbd","cbccacd","ac","cca","aaddc","dccac","ccdc","bbbbcda","ba","adbcadb","dca","abd","bdbb","ddadbad","badb","ab","aaaaa","acba","abbb"]

s = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab"
wordDict = ["a","aa","aaa","aaaa","aaaaa","aaaaaa","aaaaaaa","aaaaaaaa","aaaaaaaaa","aaaaaaaaaa"]

s = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
wordDict = ["a","aa","aaa","aaaa","aaaaa","aaaaaa","aaaaaaa","aaaaaaaa","aaaaaaaaa","aaaaaaaaaa"]


class Solution(object):
    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: bool
        """
        # 回溯,但超时,完全是靠@functools.lru_cache(None)缓存机制过的
        if not wordDict:
            return not s
        import functools
        wordDict = set(wordDict)
        min_len = min(map(len, wordDict))
        max_len = max(map(len, wordDict))
        status = [False]

        @functools.lru_cache(None)
        def backtrack(s):
            if status[0]:
                return
            if not s:
                status[0] = True
                return
            if len(s) < min_len:
                return
            for i in range(min_len, max_len + 1):
                if s[:i] in wordDict:
                    backtrack(s[i:])

        backtrack(s)
        return status[0]


solve = Solution()
print(solve.wordBreak(s, wordDict))

# ############################### 140. 单词拆分 II ###############################
s = "catsanddog"
wordDict = ["cat", "cats", "and", "sand", "dog"]
# [
#   "cats and dog",
#   "cat sand dog"
# ]

s = "pineapplepenapple"
wordDict = ["apple", "pen", "applepen", "pine", "pineapple"]
# [
#   "pine apple pen apple",
#   "pineapple pen apple",
#   "pine applepen apple"
# ]

# s = "catsandog"
# wordDict = ["cats", "dog", "sand", "and", "cat"]
# # []

# s = "acaaaaabbbdbcccdcdaadcdccacbcccabbbbcdaaaaaadb"
# wordDict = ["abbcbda","cbdaaa","b","dadaaad","dccbbbc","dccadd","ccbdbc","bbca","bacbcdd","a","bacb","cbc","adc","c","cbdbcad","cdbab","db","abbcdbd","bcb","bbdab","aa","bcadb","bacbcb","ca","dbdabdb","ccd","acbb","bdc","acbccd","d","cccdcda","dcbd","cbccacd","ac","cca","aaddc","dccac","ccdc","bbbbcda","ba","adbcadb","dca","abd","bdbb","ddadbad","badb","ab","aaaaa","acba","abbb"]

s = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
wordDict = ["a","aa","aaa","aaaa","aaaaa","aaaaaa","aaaaaaa","aaaaaaaa","aaaaaaaaa","aaaaaaaaaa"]


class Solution(object):
    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: List[str]
        """
        # if not wordDict:
        #     return []
        # import functools
        # res = []
        # wordDict = set(wordDict)
        # min_len = min(map(len, wordDict))
        # max_len = max(map(len, wordDict))
        #
        # @functools.lru_cache(None)
        # def backtrack(s, tmp):
        #     if not s:
        #         res.append(tmp.lstrip())
        #         return
        #     if len(s) < min_len:
        #         return
        #     for i in range(min_len, max_len + 1):
        #         if s[:i] in wordDict:
        #             backtrack(s[i:], tmp + ' ' + s[:i])
        #
        # backtrack(s, '')
        # return res

        import functools
        if not wordDict:return []
        wordDict = set(wordDict)
        max_len = max(map(len, wordDict))
        @functools.lru_cache(None)
        def helper(s):
            res = []
            if not s:
                res.append("")
                return res
            for i in range(len(s)):
                if i < max_len and s[:i+1] in wordDict:
                    for t in helper(s[i+1:]):
                        if not t:
                            res.append(s[:i+1])
                        else:
                            res.append(s[:i+1] + " " + t)
            return res
        return helper(s)


solve = Solution()
print(solve.wordBreak(s, wordDict))
