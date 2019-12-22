# ############################### 301. 删除无效的括号 #################################
# 删除最小数量的无效括号，使得输入的字符串有效，返回所有可能的结果。
# 说明: 输入可能包含了除 ( 和 ) 以外的字符。
s = "()())()"           # ["()()()", "(())()"]
s = "(a)())()"          # ["(a)()()", "(a())()"]
s = ")("                # [""]


class Solution:
    def removeInvalidParentheses(self, s):


solve = Solution()
print(solve.removeInvalidParentheses(s))
