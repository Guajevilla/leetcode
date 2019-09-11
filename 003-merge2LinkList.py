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

# ################################## 括号生成 #################################
