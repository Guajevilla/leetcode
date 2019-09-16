# ################################## 合并两个有序链表 #################################
#
#
# # Definition for singly-linked list.
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
# x.next.next = ListNode(4)
# # x.next.next.next = ListNode(4)
# # x.next.next.next.next = ListNode(5)
#
# y = ListNode(1)
# y.next = ListNode(3)
# y.next.next = ListNode(4)
#
# # x = None
# # y = None
#
# x = ListNode(1)
# y = None
#
#
# class Solution(object):
#     def mergeTwoLists(self, l1, l2):
#         """
#         :type l1: ListNode
#         :type l2: ListNode
#         :rtype: ListNode
#         """
#         # head = ListNode(0)
#         # p = head
#         # # head1 = ListNode(0)
#         # # head1.next = l1
#         # # head2 = ListNode(0)
#         # # head2.next = l2
#         # while l1 and l2:
#         #     n1 = l1.val
#         #     n2 = l2.val
#         #     if n1 > n2:
#         #         p.next = l2
#         #         p = p.next
#         #         if l2.next:
#         #             l2 = l2.next
#         #         else:
#         #             l2 = None
#         #             break
#         #     else:
#         #         p.next = l1
#         #         p = p.next
#         #         if l1.next:
#         #             l1 = l1.next
#         #         else:
#         #             l1 = None
#         #             break
#         # #     printList(head.next)
#         # # printList(l1)
#         # # printList(head.next)
#         # # printList(l2)
#         # if l1:
#         #     p.next = l1
#         # elif l2:
#         #     p.next = l2
#         #
#         # return head.next
#
#         # 递归
#         if l1 is None:
#             return l2
#         elif l2 is None:
#             return l1
#         elif l1.val < l2.val:
#             l1.next = self.mergeTwoLists(l1.next, l2)
#             return l1
#         else:
#             l2.next = self.mergeTwoLists(l1, l2.next)
#             return l2
#
#
# solve = Solution()
# printList(solve.mergeTwoLists(x, y))

# # ################################## 括号生成 #################################
# # [
# #   "((()))",
# #   "(()())",
# #   "(())()",
# #   "()(())",
# #   "()()()"
# # ]
# x = 3
#
#
# class Solution(object):
#     def generateParenthesis(self, n):
#         """
#         :type n: int
#         :rtype: List[str]
#         """
#         # # # 第一种想法是以入栈再出栈的方式,暂时没想好咋写
#         # # res = []
#         # # stack = ['('] * n
#         # # tmp_s = "("
#         # # list = ['('] * n + [')'] * n
#         # # while stack[-1] == "(":
#         # #
#         # # return list.
#         #
#         # # 递归,但不知为什么会有一些None的输出,主要思想是类似树的左右子树
#         # if n == 0:
#         #     return [""]
#         # lis = []
#         # cnt = 0         # 未配对的"("个数
#         # total_cnt = 0   # 一共有几个"("
#         # tmp_s = ""
#         #
#         # def addElement(total_cnt, cnt, n, tmp_s, s):
#         #     tmp_s += s
#         #     if len(tmp_s) == 2*n:
#         #         lis.append(tmp_s)
#         #         return
#         #     else:
#         #         if s == '(':
#         #             total_cnt += 1
#         #             cnt += 1
#         #         else:
#         #             cnt -= 1
#         #         if total_cnt < n:
#         #             addElement(total_cnt, cnt, n, tmp_s, '(')
#         #             if cnt > 0:
#         #                 addElement(total_cnt, cnt, n, tmp_s, ')')
#         #         else:
#         #             addElement(total_cnt, cnt, n, tmp_s, ')')
#         #
#         # addElement(total_cnt, cnt, n, tmp_s, '(')
#         # return lis
#
#         # 动态规划
#         if n == 0:
#             return []
#         total_l = []
#         total_l.append([None])    # 0组括号时记为None
#         total_l.append(["()"])    # 1组括号只有一种情况
#         for i in range(2,n+1):    # 开始计算i组括号时的括号组合
#             l = []
#             for j in range(i):    # 开始遍历 p q ，其中p+q=i-1 , j 作为索引
#                 now_list1 = total_l[j]    # p = j 时的括号组合情况
#                 now_list2 = total_l[i-1-j]    # q = (i-1) - j 时的括号组合情况
#                 for k1 in now_list1:
#                     for k2 in now_list2:
#                         if k1 == None:
#                             k1 = ""
#                         if k2 == None:
#                             k2 = ""
#                         el = "(" + k1 + ")" + k2
#                         l.append(el)    # 把所有可能的情况添加到 l 中
#             total_l.append(l)    # l这个list就是i组括号的所有情况，添加到total_l中，继续求解i=i+1的情况
#         return total_l[n]
#
#
# solve = Solution()
# print(solve.generateParenthesis(x))

# ################################## 合并K个排序链表 #################################
#
#
# # Definition for singly-linked list.
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
# x.next = ListNode(4)
# x.next.next = ListNode(4)
# x.next.next.next = ListNode(5)
# x.next.next.next.next = ListNode(5)
#
# y = ListNode(1)
# y.next = ListNode(3)
# y.next.next = ListNode(4)
#
# z = ListNode(2)
# z.next = ListNode(6)
#
# X = [x, y, z]
# # X = []
#
#
# # 暴力法:把所有列表合并再排序
# class Solution(object):
#     def mergeKLists(self, lists):
#         """
#         :type lists: List[ListNode]
#         :rtype: ListNode
#         """
#         self.nodes = []
#         head = point = ListNode(0)
#         for l in lists:
#             while l:
#                 self.nodes.append(l.val)
#                 l = l.next
#         for x in sorted(self.nodes):
#             point.next = ListNode(x)
#             point = point.next
#         return head.next
#
#
# # from Queue import PriorityQueue
# #
# #
# # # 利用优先队列
# # class Solution(object):
# #     def mergeKLists(self, lists):
# #         """
# #         :type lists: List[ListNode]
# #         :rtype: ListNode
# #         """
# #         head = point = ListNode(0)
# #         q = PriorityQueue()
# #         for l in lists:
# #             if l:
# #                 q.put((l.val, l))
# #         while not q.empty():
# #             val, node = q.get()
# #             point.next = ListNode(val)
# #             point = point.next
# #             node = node.next
# #             if node:
# #                 q.put((node.val, node))
# #         return head.next
#
#
# # # 利用之前的合并两个列表进行递归
# # class Solution(object):
# #     def mergeTwoLists(self, l1, l2):
# #         """
# #         :type l1: ListNode
# #         :type l2: ListNode
# #         :rtype: ListNode
# #         """
# #         head = ListNode(0)
# #         p = head
# #         while l1 and l2:
# #             n1 = l1.val
# #             n2 = l2.val
# #             if n1 > n2:
# #                 p.next = l2
# #                 p = p.next
# #                 if l2.next:
# #                     l2 = l2.next
# #                 else:
# #                     l2 = None
# #                     break
# #             else:
# #                 p.next = l1
# #                 p = p.next
# #                 if l1.next:
# #                     l1 = l1.next
# #                 else:
# #                     l1 = None
# #                     break
# #         if l1:
# #             p.next = l1
# #         elif l2:
# #             p.next = l2
# #
# #         return head.next
# #
# #     def onceMergeKLists(self, lists):
# #         """
# #         :type lists: List[ListNode]
# #         :rtype: ListNode
# #         """
# #         length = len(lists)
# #         i = 0
# #         j = length -1
# #         lis = []
# #         while i < j:
# #             lis.append(self.mergeTwoLists(lists[i], lists[j]))
# #             i += 1
# #             j -= 1
# #             if i == j:
# #                 lis.append(lists[i])
# #         # for item in lis:
# #         #     printList(item)
# #         return lis
# #
# #     def mergeKLists(self, lists):
# #         """
# #         :type lists: List[ListNode]
# #         :rtype: ListNode
# #         """
# #         length = len(lists)
# #         if length == 0:
# #             return ListNode(0).next
# #         while length > 1:
# #             lists = self.onceMergeKLists(lists)
# #             length = len(lists)
# #         return lists[0]
#
#
# solve = Solution()
# printList(solve.mergeKLists(X))

# ################################ 两两交换链表中的节点 ################################
#
#
# # Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None
#
#
# def printList(l: ListNode):
#     while l:
#         print("%d, " %(l.val), end='')
#         l = l.next
#     print('')
#
#
# x = ListNode(1)
# x.next = ListNode(2)
# # x.next.next = ListNode(3)
# # x.next.next.next = ListNode(4)
#
# # x = None
#
#
# # 递归
# class Solution:
#     def swapPairs(self, head: ListNode) -> ListNode:
#         if head == None or head.next == None:
#             return head
#
#         l1 = head.next
#         head.next = self.swapPairs(head.next.next)
#         l1.next = head
#
#         return l1
#
#
# # 直接交换
# # class Solution(object):
# #     def swapPairs(self, head):
# #         """
# #         :type head: ListNode
# #         :rtype: ListNode
# #         """
# #         if head:
# #             dummy = ListNode(0)
# #             dummy.next = head
# #             head = dummy
# #             # 其实这里可以少用一个指针p,但是为了便于阅读多设了一个
# #             while head.next and head.next.next:
# #                 p = head.next
# #                 f = head.next.next
# #                 p.next = f.next
# #                 f.next = p
# #                 head.next = f
# #                 head = p
# #             return dummy.next
# #         else:
# #             return None
#
#
# solve = Solution()
# printList(solve.swapPairs(x))

# # ################################ K个一组翻转链表 ################################
#
# # Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None
#
#
# def printList(l: ListNode):
#     while l:
#         print("%d, " %(l.val), end='')
#         l = l.next
#     print('')
#
#
# k = 3
# x = ListNode(1)
# x.next = ListNode(2)
# x.next.next = ListNode(3)
# x.next.next.next = ListNode(4)
# x.next.next.next.next = ListNode(5)
# # x.next.next.next.next.next = ListNode(6)
# # x.next.next.next.next.next.next = ListNode(7)
# # x.next.next.next.next.next.next.next = ListNode(8)
# # x.next.next.next.next.next.next.next.next = ListNode(9)
# # x.next.next.next.next.next.next.next.next.next = ListNode(10)
#
#
# # x = None
#
#
# # 常规方法直接交换会占用 O(k) 的存储空间,不符合要求,第一想法是递归,每次交换最外层两个
# # 或者每次将最后一个元素插到最前面
# # 不能用栈的方法,因为这题要求常数级的空间复杂度
# # class Solution(object):
# #     def reverse(self, head, k):
# #         p = head
# #         while k > 1:
# #             p = p.next
# #             k -= 1
# #         tmp = p.next.next
# #         p.next.next = head.next
# #         head.next = p.next
# #         p.next = tmp
# #
# #     def onceReverse(self, dummy, k):
# #         if k == 1:
# #             return dummy
# #         while k > 1:
# #             self.reverse(dummy, k)
# #             k -= 1
# #             dummy = dummy.next
# #         dummy = dummy.next
# #
# #         return dummy
# #
# #     def reverseKGroup(self, head, k):
# #         """
# #         :type head: ListNode
# #         :type k: int
# #         :rtype: ListNode
# #         """
# #         dummy = ListNode(0)
# #         dummy.next = head
# #         rem = dummy
# #         tmp = dummy
# #         length = 0
# #         while tmp.next:
# #             length += 1
# #             tmp = tmp.next
# #
# #         for i in range(length//k):
# #             dummy = self.onceReverse(dummy, k)
# #
# #         return rem.next
#
#
# class Solution(object):
#     # 尾插法
#     # def reverseKGroup(self, head, k):
#     #     dummy = ListNode(0)
#     #     dummy.next = head
#     #     pre = dummy
#     #     tail = dummy
#     #     while True:
#     #         count = k
#     #         while count and tail:
#     #             count -= 1
#     #             tail = tail.next
#     #         if not tail: break
#     #         head = pre.next
#     #         while pre.next != tail:
#     #             cur = pre.next          # 获取下一个元素
#     #             # pre与cur.next连接起来,此时cur(孤单)掉了出来
#     #             pre.next = cur.next
#     #             cur.next = tail.next    # 和剩余的链表连接起来
#     #             tail.next = cur         # 插在tail后面
#     #         # 改变 pre tail 的值
#     #         pre = head
#     #         tail = head
#     #     return dummy.next
#
#     # 递归
#     def reverseKGroup(self, head, k):
#         cur = head
#         count = 0
#         while cur and count != k:
#             cur = cur.next
#             count += 1
#         if count == k:
#             cur = self.reverseKGroup(cur, k)
#             while count:
#                 tmp = head.next
#                 head.next = cur
#                 cur = head
#                 head = tmp
#                 count -= 1
#             head = cur
#         return head
#
#
# solve = Solution()
# printList(solve.reverseKGroup(x, k))

# ################################ 删除排序数组中的重复项 ################################
# # 不要使用额外的数组空间,必须在原地修改输入数组并在使用 O(1) 额外空间的条件下完成。
# x = [0, 0, 0, 1, 1, 2, 3]
# # x = [0, 0, 0, 1, 1, 2]
# # x = [1, 1, 2]
# # x = []
#
#
# class Solution(object):
#     def removeDuplicates(self, nums):
#         """
#         :type nums: List[int]
#         :rtype: int
#         """
#         # # 我的想法将原数组大小改变了,其实没必要,而且时间会更长
#         # length = len(nums)
#         # if length == 0:
#         #     return 0
#         # # tmp = None
#         # # i = 0
#         # tmp = nums[0]
#         # i = 1
#         # while i < length:
#         #     if nums[i] == tmp:
#         #         length -= 1
#         #         # del操作速度比pop快
#         #         del nums[i]
#         #         # nums.pop(i)
#         #     else:
#         #         tmp = nums[i]
#         #         i += 1
#         #
#         # return length
#
#         # 用两个指针,反逻辑,每遇到 nums[i] 不等于 nums[i - 1],就说明遇到了新的不同数字,记录之
#         if not nums:
#             return 0
#         k = 1
#         for i in range(1, len(nums)):
#             if nums[i] != nums[i - 1]:
#                 nums[k] = nums[i]
#                 k += 1
#         return k
#
#
# solve = Solution()
# print(solve.removeDuplicates(x))

# # ################################# 移除元素 ##################################
# nums = [3,2,2,3]
# val = 3
#
# nums = [0,1,2,2,3,0,4,2]
# val = 2
#
#
# class Solution(object):
#     def removeElement(self, nums, val):
#         """
#         :type nums: List[int]
#         :type val: int
#         :rtype: int
#         """
#         # length = len(nums)
#         # if length == 0:
#         #     return 0
#         # i = 0
#         # while i < length:
#         #     if nums[i] == val:
#         #         length -= 1
#         #         del nums[i]
#         #     else:
#         #         i += 1
#         # return length
#
#         # # 如上,双指针,反逻辑
#         # length = len(nums)
#         # if length == 0:
#         #     return 0
#         # k = 0
#         # for i in range(length):
#         #     if nums[i] != val:
#         #         nums[k] = nums[i]
#         #         k += 1
#         # return k
#
#         # 双指针,一个从头,一个从尾开始扫,把非该数字移到前面来
#         if not nums: return 0
#         slow = 0
#         n = len(nums)
#         fast = n - 1
#         while slow < fast:
#             while slow < fast and nums[slow] != val:
#                 slow += 1
#             while slow < fast and nums[fast] == val:
#                 fast -= 1
#             nums[slow], nums[fast] = nums[fast], nums[slow]
#             slow += 1
#             fast -= 1
#         res = 0
#         for i in range(n):
#             if nums[i] == val:
#                 return res
#             res += 1
#
#
# solve = Solution()
# print(solve.removeElement(nums, val))

# # ################################# 实现 strStr() ##################################
# # 在 haystack 字符串中找出 needle 字符串出现的第一个位置 (从0开始).如果不存在,则返回 -1.
# # 当 needle 是空字符串时,返回 0
# haystack = "hello"  # 2
# needle = "ll"
#
# # haystack = "aaaaa"  # -1
# # needle = "bba"
#
# haystack = ""  # -1
# needle = "a"
#
#
# # 评论区大佬说这题暴力解法不可取,复杂度 O(m*n)
# # 用KMP算法,复杂度 O(m+n) 尽量利用之前比较过后的结果
# class Solution(object):
#     def strStr(self, haystack, needle):
#         """
#         :type haystack: str
#         :type needle: str
#         :rtype: int
#         """
#         length = len(needle)
#         if length == 0:
#             return 0
#         l = len(haystack) - length + 1
#         for i in range(l):
#             if haystack[i] == needle[0]:
#                 if haystack[i:(i+length)] == needle:
#                     return i
#
#         return -1
#
#
# solve = Solution()
# print(solve.strStr(haystack, needle))

# ################################# 两数相除 ##################################
# # 要求不使用乘法、除法和 mod 运算符。
# dividend = 10   # 3
# divisor = 5
#
# dividend = 17    # -2
# divisor = -3
#
# # dividend = 3    # -2
# # divisor = 1
#
# dividend = -2147483648
# divisor = -10
#
#
# # 解法是不断将除数翻倍
# class Solution(object):
#     def divide(self, dividend, divisor):
#         """
#         :type dividend: int
#         :type divisor: int
#         :rtype: int
#         """
#         res = 0
#         if (dividend > 0) ^ (divisor > 0):
#             flag = 1
#         else:
#             flag = 0
#         dividend, divisor = abs(dividend), abs(divisor)
#
#         while dividend - divisor >= 0:
#             i = 0
#             tmp = dividend
#             while dividend >= (divisor << i):
#                 dividend -= (divisor << i)
#                 # res += 2 ** i
#                 res += 1 << i
#                 i += 1
#
#         if flag:
#             res = -res
#         if res > (2 << 30) - 1:
#             return (2 << 30) - 1
#         elif res < 0 - (2 << 30):
#             return 0 - (2 << 30)
#         return res
#
#         # res = 0
#         # sign = 1 if dividend ^ divisor >= 0 else -1
#         # # print(sign)
#         # dividend = abs(dividend)
#         # divisor = abs(divisor)
#         # while dividend >= divisor:
#         #     tmp, i = divisor, 1
#         #     while dividend >= tmp:
#         #         dividend -= tmp
#         #         res += i
#         #         i <<= 1
#         #         tmp <<= 1
#         # res = res * sign
#         # return min(max(-2 ** 31, res), 2 ** 31 - 1)
#
#
# class Solution(object):
#     def divide(self, dividend, divisor):
#         """
#         :type dividend: int
#         :type divisor: int
#         :rtype: int
#         """
#         if (dividend > 0 and divisor > 0) or (dividend < 0 and divisor < 0):
#             tag = 1
#         else:
#             tag = -1
#         res = self.divCreas(abs(dividend), abs(divisor))
#         if tag == -1: res = 0 - res
#         if res > (2 << 30) - 1:
#             return (2 << 30) - 1
#         elif res < 0 - (2 << 30):
#             return 0 - (2 << 30)
#         return res
#
#     def divCreas(self, dividend, divisor):
#         ex, count = divisor, 0
#         t, m = 0, 1
#         if dividend < divisor:
#             return 0
#         while dividend - t - ex >= 0:
#             t += ex
#             count += m
#             m += m
#             ex += ex
#         if dividend - t > 0:
#             return count + self.divCreas(dividend - t, divisor)
#         return count
#
#
# solve = Solution()
# print(solve.divide(dividend, divisor))

# ################################# 串联所有单词的子串 ##################################
s1 = "barfoothefoobarman"    # [0, 9]
words1 = ["foo", "bar"]

s1 = "wordgoodgoodgoodbestword"  # []
words1 = ["word", "good", "best", "word"]
# words1 = []

s1 = "wordgoodgoodgoodbestword"  # [8]
words1 = ["word","good","best","good"]

s1 = "aaaaaa"
words1 = ["aaa","aaa"]

s1 = "aaaaaaaaa"  # [0, 1, 2]
words1 = ["aa","aa","aa","aa"]


s1 = "ababaab"
words1 = ["ab","ba","ba"]

s1 = "abcbaaaccaabbcababaaabccaabccccbbccbaabcbccacacacabcbbbacbcbbccabaccbbbcbaabbabbaaaacaacbcacbbaacbcbcbabbbcacbbacaacbbbcacccbbcacabbbacaccbcbaababa"
words1 = ["bcb","baa","cac","aca","cca"]
#
s1 = "cxksvbkmmlkwviteqccjhsedjmoyimskmeehhovubiszsodiqhtyaxdlktmuiggukldubzqdjiebyjkpqfpqdsepmqluwrqictcguslddphdyrsowjhbcnsqddmbvclzvqhsksxnhethvnyuxfxzsqpxvdasflscbzumssbbwuluijqhqngkfxhdahvhunjwpgkjylmwixssgizyyhifepigyenttyriebtcogbwftqmcpmcwvhcmsklyxgryxccyvhodiljbbxftjhrerurleejekacheehclvfviqxmnefzowdhswsxcbdmdfvilekzcrukityxyfwmcctwanvdoyptfnbtrnsthoepieoiklwmppkpegssgknmxpfoezilnocxsrfcebqtsdkwjfqppedmvkczjmnzcpwxiebofujyxuwgzpxotdcqnerzteyvwwseauvgoeglyctzrspmvrcjyuiraimwehdfbalretcfxxeppwdnniwnegeeotdsaixdikuodytbxasmwxzlfxzldfstaxmcflfpybdbzzewzylxwmidkjrprjjtgxwnideifjkeiqdjpogncrsmcjetsnnamlpwotftdranhdxytfnvwgkzroukdjmpucnjxscajcqtfptaujwtrguiwouzyhqulddiygjjkbesqyskjofawzisqdrqkjkvnodlwowgrbyhzruihzkezsyrvshhbreqhkbfaymsbmzaftkpvutwotnklutnnydxihcihqcidckkxwzssuogodszzmopmumwbogkhjukleukcufuqvcezxgylunxobvrsbbzkvlxbhiddnzuieyhbeimbxlpzghthksugdrjkznoomkzsiitpqhqquhraqkkbcgjhxstzhjpwtoocxirprjfmqwmhgyikgtrellftwupqldsinlzfwfrmdfvmgfwmyqsmdxhzuwpfbjprwowsvphzuelckjrkbjwejdgdbxkdhzwfnsaljjkdnxixizikigqrmwwnugsdhokxikirtuxjtfibgslozeilagywptbwhmvqwdjszgbsnjutchkdluooaompjooraljypusobvjohdklmuqyogoquaigqwxsjiryclpfjywsdgdpctpqzdivgqbwoapykiypvpuepswsybkcwzsxfbvntylibcglmeciuzojrnesqounppmwshjlgxtjzzumgzwcymlpbrjsfehxtttldfwlcsudrqpzpnbnapfbgovoucnnygadnzqrrkvkckkuanjaeodnfzbzdqpdypgmoydhiysnlehnrsnwjsloropxeeacwjomhuusuohhsqulihjrcuhvixsmdvpbefqnbmhwaodueafnjpellmhulbiqwzscfiqiuxgwomqsmxfvmmhyaqunrcdocvqjfirbiyzwmpoypwtdkcdksxzkacaeasnhbgjlgkhsaxqrvmufoyrjxqvztxdvpscszndfymaamqrhelnvleejxbiqyonpgpihdnpbcpbohuvmfkhtrncoqmgqatfjkpqnffqjutxenuqvhzoyosogeuwhpdqzvipaofjkbiooeejlfzjvrzbytxhidxkyfzavglghtuyzbhlgjwcawdardhcigmgonijvtpdokdnlmatvzxyvdymggqqmcyargmnbbqpnveahhudgtbdwzrehiuwmsyeykrbojqbexelgaomtrrqtiucspyfhxjijajxjcbpbfahfrvyimodwjgpyewhdfrphbmsfnhguhpzakalyoowzunzbjhgqyvxbkrgzyouidtinttnkkkjezjhjsqbslzuvqcvrrrzwkjkgdzsnldtlmdwgtxzewvcpxzgqqhncqzkvackmgexujtbcqcipxmgwlopdvcgndqjdvtpbzoxijamacvrzjxyvnnykpgxuxixucpvddumpvapxxizhhxeukcebjdvimucqjztpvheqivqfdpokosgyxkbipwsbqurcvltquzjcwzkzqyletteqffaubswtonxjasbvrkznljodkbhfunvzsxwvpsrdhqokjpfcceqnqgrckaheoegibceqwvvdljnwyuzcbrsrxlthlcobgwkhyqzwlubyfrvflwimnafknauacickeoteeucrodrvuobikjwxlckyeeyjoctusnawhcpyfhtcvukifgfskpspvrylvtfogfmqhcqpjlrgidopjwiunalltjwpccflhrdrvtgegznocdgnzohposakdwbgagtkxwbtrjzxkoomuuzvkjkadkkhjlpjtittewoxfpwpemdygftsqgttqfcbtrlmbefhbteijbapnfpwkkqcslwjramkuxyveeffzlpkopbevsahdskveigvivhesfcwlhdnstxhkblhtnpyfbwljegrzpysxaqihwxzrxibyvjriasqbobmskfsbdmydejkagmrdutdqevagpsjduvxgarhefihkrukzgcdcxguddvlsnuxjrxrrozvuhfgazqzhuejtlgyqdllsfiewhvqwunsdsydtqfanjmiwujpxuapcktysrqoleirwiwsabupngajcjyzdarflmgddwtradizletninfvwfgyohathrbsdhxjfsaivkjiqcyypdvniemylmrufspkbmthhvpcfanwclwtouhwavunjnhogwyhluqsphwxhjvjutfkpoipjecusmiaiijvcapujmrrxocshhexxnmgrraldklntxlxzarimkzkyceglkfjxtrrkucpeqfznqxmqqufbwrbaxhnhoyfiqwumakqsrsfhrtzhqekoxmouvdckchsufmghyyarqhyhbartebhenxylaavcjnwobeycdytthudiuudavkeljdwkdtopindjrdnudjqlftvznzbklgxvlthqmvfuklgcovysgodlhakwzmjnugifcpvqmbnzovdcqbwzsbkbcvydjhqdpakrphkeixdwuibmjxlbzwddtdgcmxhbxtvpafvleajyikkrkyvluaondwrptastvnivufiafsanengqldbfdrugonxjnqckfkfcrocwiflosufdxikbaejqthzgzcqeoxggnlexqqmkktpjbzkbfwtydtgcvyilxrrlewkwowgapvjruwubsozxjhzgfjrcalpejaazyizodihzedaytbveiwkpgesgphnajpziyyybihdpkfnghlkrhvhnzbwqkjquareyrcczjfqvkebtpmnyjwmkxkajvsfvljucnwbybsunyxjplwnusbgrlicgaieltynjwrhzlbmlzvamtphntngeyjnytrmorbxnufmfiasjwswrkdfdsljqwwrppfgggdtdkhktidcgxyxhdcmyqwqosjekomqxpmaatkvbpxhnyhwdljdbfuszfwjukctzovbjhwnxwwkwdgzppdswzkweihasjtuzoxjywwvsuhoynppfujdvwzaghcbsyxsoubmqzhitoyteqklmwoisqkaxmbpkyhztklllvwhjuapmnazjrhbhrbgffvqdfryrckdzgkjcmapzdqiuzldspjxugpxlgydliikouvsgyjgbzqxacasrjslphkdqiidsqniklbsjkymmpjmtlfkuxxlghowsyzkopvaawtlitzukijdtqppnoavyrsqptcgixgkvbxgxwcjglpzbeqqvrmtigjzbnfknowkrwqostybgnaktraokohuwstyibkvpihgeyxztvabkcldvosfcbbbuxzcajzptgxygwzbrzddbohzcbgheiiyhhchsdylmvlsukuljxrnnymqbsxfchgjoksiqqtcohwirqvdpmsfmevpyuxbbdmrpfzfvujldgtvypaqdsvqwsfwoczrhmiztjgqfqcjyvewmeoqwjiudnqrssizesazdhpjxrsxpytdektctbwzroslgbmmvnlzubitucqjalnevigrmeqfuiqblcnhrbilcqgyuwiukxafhgwtmoagxqhkvxtmabaetgcnfkjpjjurrtmdhnkgfttasmpuqpyjxbzcnirxsoojjcpspbbvuuxpimjydikbjjdwrxvlnlvwokqflrchlaywokussetdnybhxzsmkpkybbgosiwgiwcxgwradmfsmhzkguwsjhtlizbchziswmrcjifowkgitisbcrunanakocmxbxpxjicushiotpxnxrobikoixpunrhlsgcsrlwmdfusylplkgclrmcbkrwzkfkelnyeyuqdznvyamllvnymacnmvllfqymdlkilfaognmgqysbvfbjhextbkhhdftgsfqdmrttgfbwgtzdbdnijmekwntzsoikuypiridaqfyyaybbdommasyxfsyxggjchylyiqayvzywxazcolordookgmhpvstcqgcbxdzseaqbaqfqdvhjjvtqkbhhtajmhnneqoyuopxqhehkzotjmnbyqiflkoztdmzwdaqtpqkyuriwhefvtgtjqywcowyskxonxghoytovmxrtdypwgihyjdazzytkyjzxqioqbcnnbgheeyakihitnltmlmyjwyjogxeizuxbaghfeirprcienbtyqrkmrvaasgktchwdoekuobjffsmsvftlyfxqazquiankjkpxozucddjixxdtcweddevffnznpoayypyopssuxecxbfqgdwjgaglgtmvibvibngseakyaqaxuipalllsorfwksrutpcuelminzgnriklqzlcnwwbpbxzvqvohylllztyaboskadccrgppcsfgrgbhcsrcfcngynhbbbncgqexyvpbnujeamneeegljtsjhbkkcamissiqnxrarcetpsyvyehhabqjcbtgdiovawlqtfqmhxgwrgupmdxoepxistovdeqfdcvyhmloltnczhrnkqcqgzayuquxumfzoayxolozeddfkxswnuovwowqeqqaevctxasmlgnpjrwvootdjhzhxvzdnpgrmimmifavnnkxgiuwwoahxbovwqalhgcworiwyitlxdkenfakvatsbkpzaqkhwpdnillgvfrtkexyjzigcdydnqfpgrxegcroqduliogssfqdfalhglmtbrjjjiormhgckcqsswnmcfrhgcqoochrusbfcrwpyerjjhdbgsqiyhrgmhucjdtfwwmanjpopjxasceyvugvdzbpgvtsapxwlkzbvopmxonqsrqplxkqwlgfibxjquheggfdxwqwmfoewfujegzcuhhclenbbxfjfmncifbumpbiuxtadudxekcprrquqyfwksatzbpltsvnpqovltspdwgwqysgwyehsfcsitfbmdrdthygatxfrdchcuoysshlzlfifmltpcyljxrlsprjuttwpjxkbexdsenzqysidqtopmajbrvwmoudxrpaymdqsspjtjtwbomtameefzctpwxoqmpobugtnxeiizelnqeofjskkugasdoirfyucgqpfuznudzjvfxaqrnbntdiyrqrzrmbxcsdyrsuwterzdurxjskcvscpltqchrbjlgkczgyumrtqlnnufzyduauhwklddmpotbsuhsoulkmxxbtcauhwwbdsnqysdniyoasvugrgqdfneashubftdjnsblneyvcoyumsddatjhjnidueeaxjllemyrtxmxnkszfxfhqopbbxeydladunoybopwlcubooavlfddvsfxrlxuwzxrmnrpchmpliqbwtxhyckuuptldshzrfsfukwwtiogqehoxgvyigucxppahzcygwfaibzbmnjetrttzoriwnmucewldaljxqjfrkjdxsitldmlrfvoshkwnghqhszgilnbvwhvrroeuaplhmbzulxhueabybjimwjkvqhmjvqdxireuufqgcaaiadgbmoqkzafshtbemhduahquohasjcajfimryccxejpndtrpcwlcdbwtkzltbnchxpavtevyqmltffkjbvlhwkajjocmdhvbywyrctpsidnpixzlsksrwvaflcuojprhlqbqlqivtwldtkpowjftefaphugtkxcxpdndwyyrujvpvmdsxklcpntzibusbwpqcdvybupxfmobautyegcwtxvbzpvanlspqoptkhspviswclwjtafnxcqytmaiztarjpmtygkuodstqockqjznnpmgdmqehqxqgjlgrwagbuzrkdbaocobscjxqzeyqbqynegechmddnuosyogaejuiuuzuyzmzrmovutxbfchvzvnzjuzqfwyaqxwqykrqygnsznwgpddoyrnjnpzsnysdxqvyamqysdttqpcgsfwswkbjzdemdyrcpoaraqstulomcquuwroudrgcumqzkjcbxctzvlsryhdazawxrksubayy"
words1 = ["otftdranhdxytfnvwgkzroukdj","iflkoztdmzwdaqtpqkyuriwhef","lbsjkymmpjmtlfkuxxlghowsyz","cddjixxdtcweddevffnznpoayy","snjutchkdluooaompjooraljyp","fuszfwjukctzovbjhwnxwwkwdg","frmdfvmgfwmyqsmdxhzuwpfbjp","ukityxyfwmcctwanvdoyptfnbt","mhnneqoyuopxqhehkzotjmnbyq","vtgtjqywcowyskxonxghoytovm","wouzyhqulddiygjjkbesqyskjo","mfiasjwswrkdfdsljqwwrppfgg","zruihzkezsyrvshhbreqhkbfay","rsxpytdektctbwzroslgbmmvnl","jdwrxvlnlvwokqflrchlaywoku","xhnhoyfiqwumakqsrsfhrtzhqe","gtbdwzrehiuwmsyeykrbojqbex","tpcyljxrlsprjuttwpjxkbexds","tsjhbkkcamissiqnxrarcetpsy","keiqdjpogncrsmcjetsnnamlpw","rquqyfwksatzbpltsvnpqovlts","tdgcmxhbxtvpafvleajyikkrky","qvrmtigjzbnfknowkrwqostybg","vluaondwrptastvnivufiafsan","rnsthoepieoiklwmppkpegssgk","cyypdvniemylmrufspkbmthhvp","ihcihqcidckkxwzssuogodszzm","chrusbfcrwpyerjjhdbgsqiyhr","wmeoqwjiudnqrssizesazdhpjx","ommasyxfsyxggjchylyiqayvzy","kwntzsoikuypiridaqfyyaybbd","cwjomhuusuohhsqulihjrcuhvi","wxazcolordookgmhpvstcqgcbx","nusbgrlicgaieltynjwrhzlbml","xrtdypwgihyjdazzytkyjzxqio","xfvmmhyaqunrcdocvqjfirbiyz","fuklgcovysgodlhakwzmjnugif","hzhxvzdnpgrmimmifavnnkxgiu","xsmdvpbefqnbmhwaodueafnjpe","xfbvntylibcglmeciuzojrnesq","cnhrbilcqgyuwiukxafhgwtmoa","xkajvsfvljucnwbybsunyxjplw","zuieyhbeimbxlpzghthksugdrj","gbzqxacasrjslphkdqiidsqnik","jxtrrkucpeqfznqxmqqufbwrba","chziswmrcjifowkgitisbcruna","jyzdarflmgddwtradizletninf","pcktysrqoleirwiwsabupngajc","dkenfakvatsbkpzaqkhwpdnill","kbiooeejlfzjvrzbytxhidxkyf","wlopdvcgndqjdvtpbzoxijamac","xsoojjcpspbbvuuxpimjydikbj","faubswtonxjasbvrkznljodkbh","uqsphwxhjvjutfkpoipjecusmi","nawhcpyfhtcvukifgfskpspvry","xkdhzwfnsaljjkdnxixizikigq","zxgylunxobvrsbbzkvlxbhiddn","alltjwpccflhrdrvtgegznocdg","gffvqdfryrckdzgkjcmapzdqiu","hzedaytbveiwkpgesgphnajpzi","wmpoypwtdkcdksxzkacaeasnhb","hsdylmvlsukuljxrnnymqbsxfc","bbbncgqexyvpbnujeamneeeglj","bjhgqyvxbkrgzyouidtinttnkk","pyuxbbdmrpfzfvujldgtvypaqd","cfanwclwtouhwavunjnhogwyhl","plkgclrmcbkrwzkfkelnyeyuqd","ugvdzbpgvtsapxwlkzbvopmxon","msbmzaftkpvutwotnklutnnydx","pdwgwqysgwyehsfcsitfbmdrdt","elgaomtrrqtiucspyfhxjijajx","biqyonpgpihdnpbcpbohuvmfkh","llmhulbiqwzscfiqiuxgwomqsm","mpucnjxscajcqtfptaujwtrgui","gdzsnldtlmdwgtxzewvcpxzgqq","gdtdkhktidcgxyxhdcmyqwqosj","zubitucqjalnevigrmeqfuiqbl","aymdqsspjtjtwbomtameefzctp","kjezjhjsqbslzuvqcvrrrzwkjk","zavglghtuyzbhlgjwcawdardhc","fawzisqdrqkjkvnodlwowgrbyh","vrzjxyvnnykpgxuxixucpvddum","rdutdqevagpsjduvxgarhefihk","ydhiysnlehnrsnwjsloropxeea","hgjoksiqqtcohwirqvdpmsfmev","jyxuwgzpxotdcqnerzteyvwwse","sozxjhzgfjrcalpejaazyizodi","usobvjohdklmuqyogoquaigqwx","tmdhnkgfttasmpuqpyjxbzcnir","quareyrcczjfqvkebtpmnyjwmk","rmwwnugsdhokxikirtuxjtfibg","qsrqplxkqwlgfibxjquheggfdx","rukzgcdcxguddvlsnuxjrxrroz","oomuuzvkjkadkkhjlpjtittewo","wqwmfoewfujegzcuhhclenbbxf","yjogxeizuxbaghfeirprcienbt","qbwoapykiypvpuepswsybkcwzs","lvtfogfmqhcqpjlrgidopjwiun","rwowsvphzuelckjrkbjwejdgdb","jfqppedmvkczjmnzcpwxiebofu","hygatxfrdchcuoysshlzlfifml","gxqhkvxtmabaetgcnfkjpjjurr","zppdswzkweihasjtuzoxjywwvs","hgyikgtrellftwupqldsinlzfw","kckkuanjaeodnfzbzdqpdypgmo","aiijvcapujmrrxocshhexxnmgr","sjiryclpfjywsdgdpctpqzdivg","kuxyveeffzlpkopbevsahdskve","uqvhzoyosogeuwhpdqzvipaofj","gjhxstzhjpwtoocxirprjfmqwm","cwiflosufdxikbaejqthzgzcqe","qeqqaevctxasmlgnpjrwvootdj","ymggqqmcyargmnbbqpnveahhud","ekomqxpmaatkvbpxhnyhwdljdb","zvamtphntngeyjnytrmorbxnuf","uhoynppfujdvwzaghcbsyxsoub","efhbteijbapnfpwkkqcslwjram","koxmouvdckchsufmghyyarqhyh","tthudiuudavkeljdwkdtopindj","nwwbpbxzvqvohylllztyaboska","dccrgppcsfgrgbhcsrcfcngynh","qdpakrphkeixdwuibmjxlbzwdd","ftgsfqdmrttgfbwgtzdbdnijme","ounppmwshjlgxtjzzumgzwcyml","cpvqmbnzovdcqbwzsbkbcvydjh","pbrjsfehxtttldfwlcsudrqpzp","qbcnnbgheeyakihitnltmlmyjw","ztvabkcldvosfcbbbuxzcajzpt","xfpwpemdygftsqgttqfcbtrlmb","hncqzkvackmgexujtbcqcipxmg","ilfaognmgqysbvfbjhextbkhhd","hvqwunsdsydtqfanjmiwujpxua","yqrkmrvaasgktchwdoekuobjff","egeeotdsaixdikuodytbxasmwx","jfmncifbumpbiuxtadudxekcpr","slozeilagywptbwhmvqwdjszgb","kkugasdoirfyucgqpfuznudzjv","pvapxxizhhxeukcebjdvimucqj","bqurcvltquzjcwzkzqyletteqf","cbrsrxlthlcobgwkhyqzwlubyf","mqzhitoyteqklmwoisqkaxmbpk","nbnapfbgovoucnnygadnzqrrkv","ztpvheqivqfdpokosgyxkbipws","auvgoeglyctzrspmvrcjyuirai","yhmloltnczhrnkqcqgzayuquxu","funvzsxwvpsrdhqokjpfcceqnq","vuhfgazqzhuejtlgyqdllsfiew","gmhucjdtfwwmanjpopjxasceyv","vpscszndfymaamqrhelnvleejx","dzseaqbaqfqdvhjjvtqkbhhtaj","zylxwmidkjrprjjtgxwnideifj","nzohposakdwbgagtkxwbtrjzxk","igvivhesfcwlhdnstxhkblhtnp","trncoqmgqatfjkpqnffqjutxen","vwfgyohathrbsdhxjfsaivkjiq","rdnudjqlftvznzbklgxvlthqmv","kopvaawtlitzukijdtqppnoavy","raldklntxlxzarimkzkyceglkf","nakocmxbxpxjicushiotpxnxro","wxoqmpobugtnxeiizelnqeofjs","smsvftlyfxqazquiankjkpxozu","fwksrutpcuelminzgnriklqzlc","nefzowdhswsxcbdmdfvilekzcr","ibvibngseakyaqaxuipalllsor","znvyamllvnymacnmvllfqymdlk","gcvyilxrrlewkwowgapvjruwub","mwehdfbalretcfxxeppwdnniwn","wwoahxbovwqalhgcworiwyitlx","nmxpfoezilnocxsrfcebqtsdkw","engqldbfdrugonxjnqckfkfcro","grckaheoegibceqwvvdljnwyuz","jcbpbfahfrvyimodwjgpyewhdf","rvflwimnafknauacickeoteeuc","gxygwzbrzddbohzcbgheiiyhhc","wcxgwradmfsmhzkguwsjhtlizb","bikoixpunrhlsgcsrlwmdfusyl","ssetdnybhxzsmkpkybbgosiwgi","vyehhabqjcbtgdiovawlqtfqmh","opmumwbogkhjukleukcufuqvce","vjriasqbobmskfsbdmydejkagm","gjlgkhsaxqrvmufoyrjxqvztxd","yyybihdpkfnghlkrhvhnzbwqkj","kznoomkzsiitpqhqquhraqkkbc","yhztklllvwhjuapmnazjrhbhrb","jjiormhgckcqsswnmcfrhgcqoo","rphbmsfnhguhpzakalyoowzunz","igmgonijvtpdokdnlmatvzxyvd","rsqptcgixgkvbxgxwcjglpzbeq","zldspjxugpxlgydliikouvsgyj","enzqysidqtopmajbrvwmoudxrp","naktraokohuwstyibkvpihgeyx","zlfxzldfstaxmcflfpybdbzzew","mfzoayxolozeddfkxswnuovwow","rodrvuobikjwxlckyeeyjoctus","yfbwljegrzpysxaqihwxzrxiby","croqduliogssfqdfalhglmtbrj","gvfrtkexyjzigcdydnqfpgrxeg","xgwrgupmdxoepxistovdeqfdcv","oxggnlexqqmkktpjbzkbfwtydt","pyopssuxecxbfqgdwjgaglgtmv","svqwsfwoczrhmiztjgqfqcjyve","bartebhenxylaavcjnwobeycdy"]


s1 = "abababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababab"
words1 = ["ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba","ab","ba"]
# s1 = "ccdcbadacbaacddddbdaccbabdaabbcdabcbbbbacbaabdaaadacbbcdacdbabbacd"   # 13
# words1 = ["bdac","dddd"]


class Solution(object):
    def findOnce(self, s, word, num_words, len_words):
        words = word[:]
        i = 0
        while i <= len(s)-len_words:
            w = s[i:(i+len_words)]
            if w in words:
                words.remove(w)
            i += len_words

        tmp = []
        if words:
            for i in word:
                tt = s.find(i)
                if tt != -1 and tt != 0:
                    tmp.append(tt)
            if tmp:
                min_tmp = min(tmp)
            else:
                min_tmp = len_words*num_words - len_words + 1
            return min_tmp
        else:
            return -1

    def findSubstring(self, s, words):
        """
        :type s: str
        :type words: List[str]
        :rtype: List[int]
        """
        length = len(s)
        res = []
        num_words = len(words)
        if num_words == 0:
            return []
        len_words = len(words[0])

        # TODO: 递归
        i = 0
        while i <= length - num_words*len_words:
            tmp = self.findOnce(s[i:(i + num_words * len_words)], words, num_words, len_words)
            if tmp == -1:
                res.append(i)
                i += 1
            else:
                i += tmp
        return res


# # 这个按等差数列判断的方法有bug,当words中有重复关键字时需要打补丁
# class Solution(object):
#     def findOnce(self, s, num_words, len_words):
#         tmp = [-2]*num_words
#         for i in range(num_words):
#             tt = s.find(words[i])
#             while tt in tmp:
#                 tt = s.find(words[i], tt+len_words)
#
#             tmp[i] = tt
#         # 这里不需要sort,找min就够了
#         # tmp.sort()
#         tmp_min = min(tmp)
#         if tmp_min == -1:
#             return -2, 0
#         if sum(tmp) == num_words*tmp_min + num_words*(num_words-1)*len_words/2:
#             res = tmp_min
#         else:
#             return -1, tmp_min
#         return 1, res
#
#     def findSubstring(self, s, words):
#         """
#         :type s: str
#         :type words: List[str]
#         :rtype: List[int]
#         """
#         res = []
#         num_words = len(words)
#         if num_words == 0:
#             return []
#         len_words = len(words[0])
#
#         # TODO: 递归
#         i = 0
#         while len(s[i:]) >= num_words*len_words:
#             flag, tmp_i = self.findOnce(s[i:], num_words, len_words)
#             i += tmp_i
#             if flag == -2:
#                 break
#             elif flag >= 0:
#                 res.append(i)
#             i += 1
#         return res

# class Solution(object):
#     def findSubstring(self, s, words):
#         """
#         :type s: str
#         :type words: List[str]
#         :rtype: List[int]
#         """
#         if not s or not words:
#             return []
#         n = len(s)
#         k = len(words[0])
#         t = len(words) * k
#         req = {}
#         for w in words:
#             req[w] = req[w] + 1 if w in req else 1
#         ans = []
#
#         for i in range(min(k, n - t + 1)):
#             self._findSubstring(i, i, n, k, t, s, req, ans)
#         return ans
#
#     def _findSubstring(self, l, r, n, k, t, s, req, ans):
#         curr = {}
#         while r + k <= n:
#             w = s[r:r + k]
#             r += k
#             if w not in req:
#                 l = r
#                 curr.clear()
#             else:
#                 curr[w] = curr[w] + 1 if w in curr else 1
#                 while curr[w] > req[w]:
#                     curr[s[l:l + k]] -= 1
#                     l += k
#                 if r - l == t:
#                     ans.append(l)


solve = Solution()
print(solve.findSubstring(s1, words1))
