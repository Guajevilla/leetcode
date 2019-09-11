import time


###########################两数之和############################
# nums = [0,2,1,2,7,10]
#
#
# def twoSum(nums, target):
#     """
#     :type nums: List[int]
#     :type target: int
#     :rtype: List[int]
#     """
#     nums_l = nums.copy()
#     for ind1, num in enumerate(nums):
#         # nums_l = nums_l[1:]
#         nums_l.pop(0)
#         if (target - num) in nums_l:
#             ind2 = nums_l.index(target - num)
#             return [ind1, ind1 + ind2 + 1]
#
# # def twoSum(nums, target):
# #     hashmap = {}
# #     for index, num in enumerate(nums):
# #         another_num = target - num
# #         if another_num in hashmap:
# #             return [hashmap[another_num], index]
# #         hashmap[num] = index
# #     return None
#
#
# print(twoSum(nums, 4))

##########################两数相加#################################
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None
#
# l1 = ListNode(6)
# # l1.next = ListNode(4)
# # l1.next.next = ListNode(3)
#
# l2 = ListNode(5)
# # l2.next = ListNode(6)
# # l2.next.next = ListNode(4)
#
#
# # def addTwoNumbers(l1, l2):
# #     """
# #     :type l1: ListNode
# #     :type l2: ListNode
# #     :rtype: ListNode
# #
# #     """
# #     l = ListNode(0)
# #     l_ = l
# #     phead = l
# #     sum = l1.val + l2.val
# #     l.val = sum % 10
# #     while l1.next or l2.next:
# #         l_.next = ListNode(0)
# #         l = l_
# #         l_ = l.next
# #         l1 = l1.next if l1.next else ListNode(0)
# #         l2 = l2.next if l2.next else ListNode(0)
# #         l_.val = (l1.val + l2.val + sum // 10) % 10
# #         sum = l1.val + l2.val
# #     return phead
#
#
# def addTwoNumbers(self, l1, l2):
#     """
#     :type l1: ListNode
#     :type l2: ListNode
#     :rtype: ListNode
#     """
#     l = ListNode(0)
#     phead = l
#     sum = 0
#
#     while l1 or l2:
#         x = l1.val if l1 else 0
#         y = l2.val if l2 else 0
#         sum = x + y + sum
#         l.next = ListNode(sum % 10)
#         l = l.next
#         if l1 != None: l1 = l1.next
#         if l2 != None: l2 = l2.next
#         sum = sum // 10
#         if sum >= 1:
#             l.next = ListNode(1)
#     return phead.next
#
#
# def printList(l: ListNode):
#     while l:
#         print("%d, " %(l.val), end = '')
#         l = l.next
#     print('')
#
#
# printList(addTwoNumbers(l1, l2))


##########################无重复字符的最长子串##########################
# s = "abcabcbb"
# s = "abcaef"
# # s = "abcda"
#
#
# # def lengthOfLongestSubstring(s):
# #     """
# #     :type s: str
# #     :rtype: int
# #     """
# #     length = 0
# #     l = []
# #     for ss in s:
# #         while ss in l:
# #             l = l[1:]
# #         l.append(ss)
# #         if len(l) > length:
# #             length = len(l)
# #
# #     return length
#
#
# def lengthOfLongestSubstring(s):
#     """
#     :type s: str
#     :rtype: int
#     """
#     length = 0
#     l = []
#     for ss in s:
#         if ss in l:
#             index = l.index(ss)     # 利用字典的index方法比原想法while循环快！
#             l = l[index + 1:]
#         l.append(ss)
#         if len(l) > length:
#             length = len(l)
#
#     return length
#
#
# print(lengthOfLongestSubstring(s))


##########################寻找两个有序数组的中位数##########################
nums1 = [1, 3]  # ans = 3
nums2 = [2, 5, 7]

nums1 = [1, 3, 4]  # ans = 4.5
nums2 = [2, 5, 6, 7, 8]

# nums1 = [1, 9]  # ans = 9
# nums2 = [2, 10, 11]

# nums1 = [1, 2]  # ans = 2.5
# nums2 = [3, 4]

# nums1 = [99, 100, 101]  # ans = 7.5
# nums2 = [2, 5, 6, 7, 8]

# nums1 = [ ]  # ans = 2.5
# nums2 = [1, 2, 3]

nums1 = [100001]  # ans = 2.5
nums2 = [100000]

nums1 = [0, 0]  # ans = 2.5
nums2 = [0, 0]
#
# nums1 = [2,3,4,5,6,7,8]  # ans = 2.5
# nums2 = [1]

############补丁太多。想法是找第k小的数，每次丢弃k//2个数
# def findMedianSortedArrays(nums1, nums2):
#     m, n = len(nums1), len(nums2)
#     odd_flag = 0
#     if (m + n) % 2:
#         mid = (m + n) // 2
#         odd_flag = 1
#     else:
#         mid = (m + n - 1) // 2
#     if m*n == 0:
#         if m:
#             if odd_flag:
#                 return nums1[mid]
#             else:
#                 return (nums1[mid] + nums1[mid+1]) / 2.0
#         else:
#             if odd_flag:
#                 return nums2[mid]
#             else:
#                 return (nums2[mid] + nums2[mid+1]) / 2.0
#     if mid == 0:
#         return (nums1[0] + nums2[0]) / 2.0
#     return dropKmin(nums1, nums2, mid, odd_flag)
#
#

# def dropKmin(nums1, nums2, k, odd_flag):
#     m, n = len(nums1), len(nums2)
#     if k == 1:
#         drop = 1
#     else:
#         drop = k // 2
#
#     if m < drop:
#         num1 = nums1[-1]
#         num2 = nums2[drop - 1]
#     elif n < drop:
#         num1 = nums1[drop - 1]
#         num2 = nums2[-1]
#     else:
#         num1 = nums1[drop - 1]
#         num2 = nums2[drop - 1]
#     if num1 > num2:
#         nums2 = nums2[drop:]
#     else:
#         nums1 = nums1[drop:]
#
#     # 有一个数组空了怎么办
#     if len(nums1)*len(nums2) == 0:
#         if len(nums1):
#             drop = n
#             if odd_flag:
#                 return nums1[k-drop]
#             else:
#                 return (nums1[k-drop] + nums1[k-drop+1]) / 2.0
#         else:
#             drop = m
#             if odd_flag:
#                 return nums2[k-drop]
#             else:
#                 return (nums2[k-drop] + nums2[k-drop+1]) / 2.0
#
#     if k-drop:
#         return dropKmin(nums1, nums2, k-drop, odd_flag)
#     else:
#         if odd_flag:
#             return min(nums1[0], nums2[0])
#         else:
#             tmp1 = dropKmin(nums1, nums2, 1, 1)
#             return (tmp1 + min(nums1[0], nums2[0])) / 2.0


# findMedianSortedArrays(nums1, nums2)

###########################最长回文串##############################
s = "babad"
# s = "babadabab"
s = "abacdgfdcaba"
# s = "a"


######## 给跪，当s为超长回文时超出时间限制
# def longestPalindrome(s):
#     """
#     :type s: str
#     :rtype: str
#     """
#     # r = s[::-1]
#     total_len = len(s)
#     length = 0
#     start = None
#     for i, ss in enumerate(s):      # i指第i元素,ind指i后面与其元素指相等的元素
#         ind = i
#         while (ind+1) <= len(s):
#             if ss in s[(ind+1):]:
#                 tmp = s[(ind+1):].index(ss)
#                 ind = ind + tmp + 1   # 在原长下的索引
#                 tmp_ind = ind
#                 tmp_i = i
#                 while s[tmp_i] == s[tmp_ind]:
#                     tmp_i = tmp_i + 1
#                     tmp_ind = tmp_ind - 1
#                     if tmp_i >= tmp_ind:         # 说明是回文
#                         if length < ind - i + 1:
#                             length = ind - i + 1
#                             start = i
#                         break
#                 # tmp_i = tmp_ind
#                 # tmp_i = i
#             else: break
#
#     if length:
#         return s[start:(start+length)]
#     elif total_len:
#         return s[0]
#     else:
#         return ""

# s = "abcba"

# 动态规划 dp[j][i] 表示字符串从 j 到 i 是否是为回文串
# 即当s[j] == s[i]如果dp[j+1][i-1]也是回文串，那么字符串从j到i也是回文串，即 dp[j][i] 为真。
def longestPalindrome(s: str) -> str:
    if not s:
        return ""
    res = ""
    n = len(s)
    dp = [[0] * n for _ in range(n)]
    max_len = float("-inf")
    for i in range(n):
        for j in range(i + 1):
            # if s[i] == s[j] and (i - j <= 2 or dp[j + 1][i - 1]):
            if s[i] == s[j] and (i - j <= 2 or dp[j + 1][i - 1]):
                dp[j][i] = 1
            if dp[j][i] and max_len < i + 1 - j:
                res = s[j : i + 1]
                max_len = i + 1 - j
    return res

# # 最优
# def longestPalindrome(s):
#     if not s:
#         return ""
#     max_len = 1
#     n = len(s)
#     start = 0
#     for i in range(1, n):
#         even = s[i - max_len:i + 1]
#         odd = s[i - max_len - 1:i + 1]
#         # print(even,odd)
#         if i - max_len - 1 >= 0 and odd == odd[::-1]:
#             start = i - max_len - 1
#             max_len += 2
#         elif i - max_len >= 0 and even == even[::-1]:
#             start = i - max_len
#             max_len += 1
#
#     return s[start: start + max_len]

s = "cbcdcbedcbc"
# s = "dssa"


# 以自写第一版基础，倒着判断
# def longestPalindrome(s):
#     """
#     :type s: str
#     :rtype: str
#     """
#     # r = s[::-1]
#     total_len = len(s)
#     length = 1
#     start = 0
#     for i, ss in enumerate(s):      # i指第i元素,ind指i后面与其元素指相等的元素
#         ind = total_len
#         if length >= total_len-i:
#             break
#         # while (ind-1) >= i:
#         #     if ss in s[(i+1):ind]:
#         #         tmp = s[(i+1):ind][::-1].index(ss)
#         #         ind = ind - tmp - 1               # 在原长下的索引
#         #         if s[i:(ind+1)]==s[i:(ind+1)][::-1]:
#         #             if length < ind - i + 1:
#         #                 length = ind - i + 1
#         #                 start = i
#         #                 break
#         #     else: break
#
#         while ss in s[(i+1):ind]:
#             tmp = s[(i+1):ind][::-1].index(ss)
#             ind = ind - tmp - 1               # 在原长下的索引
#             if s[i:(ind+1)] == s[i:(ind+1)][::-1]:
#                 if length < ind - i + 1:
#                     length = ind - i + 1
#                     start = i
#                     break
#
#     if length:
#         return s[start:(start+length)]
#     else:
#         return ""


print(longestPalindrome(s))
