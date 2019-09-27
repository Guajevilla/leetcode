import time

# ################################# 41.缺失的第一个正数 ##################################
# # 要求时间复杂度应为O(n)，并且只能使用常数级别的空间
# num1 = [1,2,0]      # 3
# num1 = [3,4,-1,1]      # 2
# # num1 = [7,8,9,11,12]      # 1
# # num1 = [0, -1]      # 1
# # num1 = [1]      # 2
#
#
# # 列表的in时间复杂度为 O(n)
# # 集合(set,类似于dict)的in时间复杂度为 O(1),但是在转化成集合的时候需要 O(n)空间复杂度
# # 错误示范:
# class Solution(object):
#     # # 时间复杂度大于 O(n)
#     # def firstMissingPositive(self, nums):
#     #     """
#     #     :type nums: List[int]
#     #     :rtype: int
#     #     .
#     #     """
#     #     for i in range(1, len(nums)+2):
#     #         if i not in nums:
#     #             return i
#     #     return 1
#
#     # O(n)空间复杂度
#     # def firstMissingPositive(self, nums):
#         # length = len(nums) + 1
#         # dic = {}
#         #
#         # for i in nums:
#         #     if 0 < i <= length:
#         #         dic[i] = 1
#         #
#         # cnt = 1
#         # while 1:
#         #     try:
#         #         dic[cnt]
#         #         cnt += 1
#         #     except:
#         #         return cnt
#
#     def firstMissingPositive(self, nums):
#         n = len(nums)
#         for i in range(n):
#             while 1 <= nums[i] <= n and nums[nums[i] - 1] != nums[i]:
#                 nums[nums[i] - 1], nums[i] = nums[i], nums[nums[i] - 1]
#         # print(nums)
#         i = 0
#         while i < n and i + 1 == nums[i]:
#             i += 1
#         return i + 1
#
#
#     # # 预处理去掉有符号数,然后利用第i个数的符号表示第i个数是否存在
#     # def firstMissingPositive(self, nums):
#     #     n = len(nums)
#     #
#     #     # 基本情况
#     #     if 1 not in nums:
#     #         return 1
#     #
#     #     # nums = [1]
#     #     if n == 1:
#     #         return 2
#     #
#     #     # 用 1 替换负数, 0和大于 n 的数
#     #     # 在转换以后 nums 只会包含正数
#     #     for i in range(n):
#     #         if nums[i] <= 0 or nums[i] > n:
#     #             nums[i] = 1
#     #
#     #     # 使用索引和数字符号作为检查器
#     #     # 例如，如果 nums[1] 是负数表示在数组中出现了数字 `1`
#     #     # 如果 nums[2] 是正数 表示数字 2 没有出现
#     #     for i in range(n):
#     #         a = abs(nums[i])
#     #         # 如果发现了一个数字 a - 改变第 a 个元素的符号
#     #         # 注意重复元素只需操作一次
#     #         if a == n:
#     #             nums[0] = - abs(nums[0])
#     #         else:
#     #             nums[a] = - abs(nums[a])
#     #
#     #     # 现在第一个正数的下标就是第一个缺失的数
#     #     for i in range(1, n):
#     #         if nums[i] > 0:
#     #             return i
#     #
#     #     if nums[0] > 0:
#     #         return n
#     #
#     #     return n + 1
#
#
# solve = Solution()
# print(solve.firstMissingPositive(num1))

# ################################# 42.接雨水 ##################################
# num1 = [0,1,0,2,1,0,1,3,2,1,2,1]    # 6
#
# # num1 = [5,2,1,2,1,5]                # 14
#
# # num1 = [0,0,1,2,3]    # 6
#
# # num1 = [2,1,0,1,3]
# num1 = [5,5,1,7,1,1,5,2,7,6]   # 23
# # num1 = [4,2,0,3,2,5]   # 9
# # num1 = [5,5,4,7,8,2,6,9,4,5]
# num1 = [9,6,8,8,5,6,3]   # 3
#
# num1 = [8,8,1,5,6,2,5,3,3,9]   # 31
#
#
# class Solution(object):
#     # 形参需要保证第一个和最后一个数是边界,而且不能是0
#     def onceTrap(self, height):
#         res = 0
#         length = len(height)
#         min_border = min(height[0], height[-1])
#         for ind in range(1, length):
#             if height[ind] < min_border:
#                 res += min_border - height[ind]
#         return res
#
#     def trap(self, height):
#         """
#         :type height: List[int]
#         :rtype: int
#         """
#         length = len(height)
#         if length < 3:
#             return 0
#         res = 0
#         border = []
#         # TODO 找边界
#         for ind in range(len(height)):
#             if height[ind] == height[ind+1]:
#                 height = height[1:]
#             else:
#                 break
#
#         ind = 0
#         while ind < len(height):
#             tmp = len(border)
#             if ind == 0:
#                 if height[ind] > height[ind+1]:
#                     border.append(ind)
#             elif ind == len(height)-1:
#                 if height[ind] > height[ind-1]:
#                     border.append(ind)
#             elif height[ind] >= height[ind-1] and height[ind] >= height[ind+1]:
#                 border.append(ind)
#
#             ind += 1
#             if len(border) > 2 and len(border) != tmp:
#                 if height[border[-2]] <= height[border[-1]] and height[border[-2]] <= height[border[-3]]:
#                     border.pop(-2)
#
#         print(border)
#         for ind in range(len(border)-1):
#             res += self.onceTrap(height[border[ind]:border[ind+1]+1])
#
#         return res
#
#
# class Solution(object):
#     # 双指针
#     def trap(self, height):
#         if not height: return 0
#         left = 0
#         right = len(height) - 1
#         res = 0
#         # 记录左右边最大值
#         left_max = height[left]
#         right_max = height[right]
#         while left < right:
#             if height[left] < height[right]:
#                 if left_max > height[left]:
#                     res += left_max - height[left]
#                 else:
#                     left_max = height[left]
#                 left += 1
#             else:
#                 if right_max > height[right]:
#                     res += right_max - height[right]
#                 else:
#                     right_max = height[right]
#                 right -= 1
#         return res
#
#     # # 栈
#     # def trap(self, height):
#     #     if not height: return 0
#     #     n = len(height)
#     #     stack = []
#     #     res = 0
#     #     for i in range(n):
#     #         #print(stack)
#     #         while stack and height[stack[-1]] < height[i]:
#     #             tmp = stack.pop()
#     #             if not stack: break
#     #             res += (min(height[i], height[stack[-1]]) - height[tmp]) * (i-stack[-1] - 1)
#     #         stack.append(i)
#     #     return res
#
# # # 挖地法
# # class Solution(object):
# #     def preprocess(self, height):
# #         for ele in height:
# #             if ele == 0:
# #                 height = height[1:]
# #             else:
# #                 break
# #         i = len(height) - 1
# #         while i >= 0:
# #             if height[i] == 0:
# #                 height.pop()
# #                 i -= 1
# #             else:
# #                 break
# #         return height
# #
# #     def dig(self, height):
# #         res = 0
# #
# #         min_border = min(height[0], height[-1])
# #         for ind in range(len(height)):
# #             height[ind] -= min_border
# #             if height[ind] < 0:
# #                 res -= height[ind]
# #                 height[ind] = 0
# #         return height, res
# #
# #     def trap(self, height):
# #         res = 0
# #
# #         height = self.preprocess(height)
# #         while len(height) >= 3:
# #             height, tmp = self.dig(height)
# #             res += tmp
# #             height = self.preprocess(height)
# #
# #         return res
#
#
# solve = Solution()
# print(solve.trap(num1))
#
# ################################ 43.字符串相乘 #################################
# # 个人理解 难点在于怎么保证不溢出 所以计算结果必须保存为string类型, 模拟竖式乘法
# num1 = "2"      # "6"
# num2 = "3"
#
# num1 = "2464"      # "6"
# num2 = "3"
#
# num1 = "123"        # "56088"
# num2 = "456"
#
#
# class Solution(object):
#     # def add(self, num1, num2):
#     #     dic = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}
#     #     if len(num2)*len(num1) == 0:
#     #         if len(num1):
#     #             return num1
#     #         else:
#     #             return num2
#     #     i = 1
#     #     mem = 0
#     #     res = ''
#     #     while i <= min(len(num2), len(num1)):
#     #         tmp = dic[num1[-i]] + dic[num2[-i]]
#     #         tmp += mem
#     #         mem = tmp // 10
#     #         tmp = tmp % 10
#     #         res = str(tmp) + res
#     #         i += 1
#     #
#     #     if len(num2) >= i:
#     #         num2 = num2[:1-i]
#     #         num2 = str(int(num2) + mem)
#     #         res = num2 + res
#     #     elif len(num1) >= i:
#     #         num1 = num1[:1-i]
#     #         num1 = str(int(num1) + mem)
#     #         res = num1 + res
#     #     elif mem != 0:
#     #         res = str(mem) + res
#     #     return res
#     #
#     # def multiply(self, num1, num2):
#     #     """
#     #     :type num1: str
#     #     :type num2: str
#     #     :rtype: str
#     #     """
#     #     if num1 == '0' or num2 == '0':
#     #         return '0'
#     #     res = ''
#     #     dic = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}
#     #     cnt = 0
#     #     for s1 in num1[::-1]:
#     #         mem = 0
#     #         tmp_str = ''
#     #         for s2 in num2[::-1]:
#     #             tmp = dic[s1] * dic[s2]
#     #             tmp += mem
#     #             mem = tmp // 10
#     #             tmp = tmp % 10
#     #             tmp_str = str(tmp) + tmp_str
#     #         if mem != 0:
#     #             tmp_str = str(mem) + tmp_str
#     #         # print(tmp_str)
#     #         tmp_str += '0'*cnt
#     #         res = self.add(res, tmp_str)
#     #         cnt += 1
#     #     return res
#
#
#     def multiply(self, num1, num2):
#         """
#         :type num1: str
#         :type num2: str
#         :rtype: str
#         """
#         if num1 == "0" or num2 == "0":
#             return "0"
#         num2 = num2[::-1]
#         num1 = num1[::-1]
#         lenNum = len(num1) + len(num2) # 保存最终最大的数字
#         returnNum = [0 for c in range(lenNum)] # 用list先存储
#         for index2 in range(len(num2)):
#             multiplier2 = int(num2[index2]) # 就直接按照顺序放，最后再反过来！
#             for index1 in range(len(num1)):
#                 multiplier1 = int(num1[index1])
#                 temp = multiplier2 * multiplier1 + returnNum[index1 + index2]
#                 if temp >= 10:
#                     returnNum[index1 + index2] = temp % 10
#                     returnNum[index1 + index2 + 1] += int(temp / 10)
#                 else:
#                     returnNum[index1 + index2] = temp
#         returnNum = returnNum[::-1]
#         while returnNum and returnNum[0] == 0:
#             del returnNum[0]
#         returnNum = [str(c) for c in returnNum]
#         return ''.join(returnNum)
#
#
# solve = Solution()
# print(solve.multiply(num1, num2))

# ################################ 44.通配符匹配 #################################
# '?' 可以匹配任何单个字符。
# '*' 可以匹配任意字符串（包括空字符串）。
s = "aa"
p = "*"     # true

s = "adceb"
p = "*a*b"     # true

s = "acdcb"
p = "a*c?b"     # false

# s = "accdcb"
# p = "ac*b"     # true

# s = "accdcba"
# p = "*cba"     # true

# s = "accccafgcb"
# p = "*cb"     # true

# s = "aaaa"
# p = "***aa"

# s = "babaaababaabababbbbbbaabaabbabababbaababbaaabbbaaab"
# p = "***bba**a*bbba**aab**b"

# s = "abbbb"
# p = "?*b**"

# s = "accbdcbd"
# p = "ac*bd"     # true
#
# s = "babbbbaabababaabbababaababaabbaabababbaaababbababaaaaaabbabaaaabababbabbababbbaaaababbbabbbbbbbbbbaabbb"
# p = "b**bb**a**bba*b**a*bbb**aba***babbb*aa****aabb*bbb***a"


class Solution(object):
    def isMatch(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: bool
        """
        # # 动态规划
        # 初始化要注意下面这句,如果按这样写,每一行指向的是同一内存,改变一个,一列全都会改变!!!
        # dp = [[False]*(len(s)+1)] * (len(p)+1)
        dp = [[False] * (len(s) + 1) for _ in range(len(p) + 1)]
        dp[0][0] = True
        for i, ele in enumerate(p):
            if ele != '*':
                break
            dp[i+1][0] = True
        for i in range(len(p)):
            for j in range(len(s)):
                if s[j] == p[i] or p[i] == '?':
                    dp[i+1][j+1] = dp[i][j]
                elif p[i] == '*' and (dp[i+1][j] or dp[i][j+1]):
                    dp[i+1][j+1] = True
        print(dp)
        return dp[-1][-1]


        # i = 0
        # j = 0
        # start = -1
        # match = 0
        # while i < len(s):
        #     # 一对一匹配,匹配成功一起移
        #     if j < len(p) and (s[i] == p[j] or p[j] == "?"):
        #         i += 1
        #         j += 1
        #     # 记录p的"*"的位置,还有s的位置
        #     elif j < len(p) and p[j] == "*":
        #         start = j
        #         match = i
        #         j += 1
        #     # j 回到 记录的下一个位置
        #     # match 更新下一个位置
        #     # 这不代表用*匹配一个字符
        #     elif start != -1:
        #         j = start + 1
        #         match += 1
        #         i = match
        #     else:
        #         return False
        #  # 将多余的 * 直接匹配空串
        # return all(x == "*" for x in p[j:])


    # # 双指针
    # def isMatch(self, s, p):
    #     si,pi,pr,sr=0,0,-1,-1
    #     while si < len(s):
    #         if pi < len(p) and p[pi] == '*':
    #             pi += 1
    #             pr = pi
    #             sr = si
    #         elif pi < len(p) and (p[pi] == '?' or p[pi] == s[si]):
    #             pi += 1
    #             si += 1
    #         elif pr != -1:
    #             pi = pr
    #             sr += 1
    #             si = sr
    #         else:
    #             return False
    #
    #     while pi < len(p) and p[pi] == '*':
    #         pi += 1
    #
    #     return pi == len(p)


solve = Solution()
print(solve.isMatch(s, p))

# # ################################ 45.跳跃游戏 II #################################
# num = [2,3,1,1,4]       # 2
# num = [10,9,8,7,6,5,4,3,2,1,1,0]
# # num = [5,2,1,1,1,3,4]       # 2
# # num = [7,0,9,6,9,6,1,7,9,0,1,2,9,0,3]       # 2
#
#
# # 贪心算法 每次跳到能达到最远的子域
# class Solution(object):
#     def jump(self, nums):
#         """
#         :type nums: List[int]
#         :rtype: int
#         """
#         cnt = 0
#         length = len(nums)
#         i = 0
#         while i < length - 1:
#             cnt += 1
#             if i+nums[i] < length - 1:
#                 tmp = nums[i+1:(i+nums[i]+1)]
#                 tmp_max = 0
#                 ind_max = -1
#                 for ind, ele in enumerate(tmp):
#                     ele += ind
#                     if ele >= tmp_max:
#                         tmp_max = ele
#                         ind_max = ind
#                 i = i + ind_max + 1
#             else:
#                 break
#
#         return cnt
#
# solve = Solution()
# print(solve.jump(num))

# # ################################ 46.全排列 #################################
# num = [1,2,3,4,5,6,7,8]
# num = [1,2,3]
#
#
# # 想到的方法还是递归,如果有3个数,已知之前2个数的排列后,在每个空插入新元素,就得到3个数的排列
# # class Solution(object):
# #     def insert(self, num, nums_lis):
# #         res = []
# #         length = len(nums_lis[0])
# #         for lis in nums_lis:
# #             i = 0
# #             while i <= length:
# #                 tmp = lis[:]
# #                 tmp.insert(i, num)
# #                 res.append(tmp)
# #                 i += 1
# #         return res
# #
# #     def permute(self, nums):
# #         """
# #         :type nums: List[int]
# #         :rtype: List[List[int]]
# #         """
# #         length = len(nums)
# #         if length == 0:
# #             return []
# #         res = [[nums[0]]]
# #         i = 1
# #         while i < length:
# #             res = self.insert(nums[i], res)
# #             i += 1
# #         return res
#
#
# # 回溯法 结合树的概念
# class Solution(object):
#     def permute(self, nums):
#         """
#         :type nums: List[int]
#         :rtype: List[List[int]]
#         """
#         res = []
#
#         def backtrack(nums, temp):
#             if not nums:
#                 res.append(temp)
#
#             for i in range(len(nums)):
#                 # nums[:i] + nums[i+1:]的意义在于从原nums数组中将已经出现过的数剔除
#                 backtrack(nums[:i] + nums[i+1:], temp + [nums[i]])
#
#         backtrack(nums, [])
#         return res
#
#
# solve = Solution()
# ans = solve.permute(num)
# print(ans)
# print(len(ans))

# # ################################ 47.全排列 II #################################
# # 有重复
# num = [1,1,2,3]
# num = [1,1,2,2]
# # num = [1,2,3]
#
#
# class Solution(object):
#     # def insert(self, num, nums_lis):
#     #     res = []
#     #     length = len(nums_lis[0])
#     #     for lis in nums_lis:
#     #         i = 0
#     #         while i <= length:
#     #             tmp = lis[:]
#     #             if i == length or tmp[i] != num:
#     #                 tmp.insert(i, num)
#     #                 if tmp not in res:
#     #                     res.append(tmp)
#     #             i += 1
#     #     return res
#     #
#     # def permuteUnique(self, nums):
#     #     """
#     #     :type nums: List[int]
#     #     :rtype: List[List[int]]
#     #     """
#     #     nums.sort()
#     #     length = len(nums)
#     #     if length == 0:
#     #         return []
#     #     res = [[nums[0]]]
#     #     i = 1
#     #     while i < length:
#     #         res = self.insert(nums[i], res)
#     #         i += 1
#     #     return res
#
#     def permuteUnique(self, nums):
#         """
#         :type nums: List[int]
#         :rtype: List[List[int]]
#         """
#         nums.sort()
#         res = []
#
#         def backtrack(nums, temp):
#             if not nums:
#                 res.append(temp)
#
#             for i in range(len(nums)):
#                 # 剪枝
#                 if i > 0 and nums[i] == nums[i - 1]:
#                     continue
#                 # nums[:i] + nums[i+1:]的意义在于从原nums数组中将已经出现过的数剔除
#                 backtrack(nums[:i] + nums[i+1:], temp + [nums[i]])
#
#         backtrack(nums, [])
#         return res
#
#
# solve = Solution()
# ans = solve.permuteUnique(num)
# print(ans)
# print(len(ans))

# # ################################ 48.旋转图像 #################################
# # n*n
# matrix = [
#   [1,2,3],
#   [4,5,6],
#   [7,8,9]]
# # [
# #   [7,4,1],
# #   [8,5,2],
# #   [9,6,3]
# # ]
#
# # matrix = [
# #   [ 5, 1, 9,11],
# #   [ 2, 4, 8,10],
# #   [13, 3, 6, 7],
# #   [15,14,12,16]]
# # [
# #   [15,13, 2, 5],
# #   [14, 3, 4, 1],
# #   [12, 6, 8, 9],
# #   [16, 7,10,11]
# # ]
# # matrix = [
# #   [1,2],
# #   [4,5]]
#
# # matrix = [[1]]
#
#
# class Solution(object):
#     def rotate(self, matrix):
#         """
#         :type matrix: List[List[int]]
#         :rtype: None Do not return anything, modify matrix in-place instead.
#         """
#         # 先转置后交换
#         # length = len(matrix)
#         # for i in range(length):
#         #     for j in range(i, length):
#         #         matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
#         # # for ind, row in enumerate(matrix):
#         # #     matrix[ind] = row[::-1]
#         # for ind in range(length):
#         #     matrix[ind].reverse()
#
#         # # 先交换后转置
#         # n = len(matrix)
#         # matrix.reverse()
#         # #print(matrix)
#         # for i in range(0, n):
#         #     for j in range(i+1, n):
#         #         matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
#
#         # 四个一交换
#         n = len(matrix)
#         for i in range(n // 2):
#             for j in range(i, n-i-1):
#                 matrix[i][j], matrix[j][n-i-1], matrix[n-i-1][n-j-1], matrix[n-j-1][i] = \
#                     matrix[n-j-1][i], matrix[i][j], matrix[j][n-i-1], matrix[n-i-1][n-j-1]
#
#
# solve = Solution()
# solve.rotate(matrix)
# print(matrix)

# # ################################ 49 字母异位词分组 #################################
# lis = ["tea", "seat", "tan", "ate", "nat", "bat"]
# # lis = ["eat", "tea", "tan", "ate", "nat", "bat"]
# # [
# #   ["ate","eat","tea"],
# #   ["nat","tan"],
# #   ["bat"]
# # ]
# lis = ["","b"]
# # lis = ["",""]
# # lis = ["tea","and","ace","ad","eat","dans"]
# lis = ["tee", "eat", "tan", "ate", "nat", "bat"]
#
#
# # 本题关键在于hash表选什么作为键值,可以是排序后的字符串,质数乘积或者统计字母出现的次数“a2b3c0....”
# class Solution(object):
#     def groupAnagrams(self, strs):
#         """
#         :type strs: List[str]
#         :rtype: List[List[str]]
#         """
#         # sorted排序法
#         dic = {}
#         for i in range(len(strs)):
#             rem = "".join(sorted(strs[i]))
#             if rem not in dic:
#                 dic[rem] = [strs[i]]
#             else:
#                 dic[rem].append(strs[i])
#         return list(dic.values())
#
#     # # 质数相乘法
#     # def groupAnagrams(self, strs):
#     #     dic = {'a':2,'b':3,'c':5,'d':7,'e':11,'f':13,'g':17,'h':19,'i':23,'j':29,'k':31,'l':37,'m':41,
#     #     'n':43,'o':47,'p':53,'q':59,'r':61,'s':67,'t':71,'u':73,'v':79,'w':83,'x':89,'y':97,'z':101}
#     #     # 下面写的两种解法不同在于字典的初始化,一个用defaultdict初始化值为列表的字典,一个判断后初始化
#     #     # from collections import defaultdict
#     #     # dic_rem = defaultdict(list)
#     #     # for s in strs:
#     #     #     tmp = 1
#     #     #     for ele in s:
#     #     #         tmp = tmp * dic[ele]
#     #     #     dic_rem[tmp].append(s)
#     #     #
#     #     # res = list(dic_rem.values())
#     #     # return res
#     #
#     #     dic_rem = {}
#     #     for s in strs:
#     #         tmp = 1
#     #         for ele in s:
#     #             tmp = tmp * dic[ele]
#     #         if tmp not in dic_rem:
#     #             dic_rem[tmp] = [s]
#     #         else:
#     #             dic_rem[tmp].append(s)
#     #
#     #     res = list(dic_rem.values())
#     #     return res
#
#
# solve = Solution()
# print(solve.groupAnagrams(lis))

# # ################################ 50 Pow(x, n) #################################
# # -100.0 < x < 100.0
# # n 是 32 位有符号整数，其数值范围是 [−2^31, 2^31 − 1] [-2147483648, 2147483647]
# x, n = 2.00000, 10
# x, n = 2.10000, 3       # 9.26100
# x, n = 2.00000, -2      # 0.25000
# x, n = 2.00000, 1000      # 0.25000
#
#
# # 二分法 幂指数每次二分
# class Solution(object):
#     def myPow(self, x, n):
#         """
#         :type x: float
#         :type n: int
#         :rtype: float
#         """
#         # if n == 0:
#         #     return 1
#         # sign = n < 0
#         # n = abs(n)
#         # res = 1.0
#         # while n > 1:
#         #     if n % 2:
#         #         res *= x
#         #         n -= 1
#         #     else:
#         #         x = x * x
#         #         n = n >> 1
#         #
#         # res *= x
#         # if sign:
#         #     res = 1.0 / res
#         # return res
#
#         if n < 0:
#             x = 1 / x
#             n = -n
#
#         res = 1
#         while n:
#             if n & 1:
#                 res *= x
#             x *= x
#             n >>= 1
#         return res
#
#
# solve = Solution()
# print(solve.myPow(x, n))
