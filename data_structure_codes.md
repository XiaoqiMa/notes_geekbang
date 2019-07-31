[TOC]

```python
# 牛客网
lines = sys.stdin.readlines()
n = int(lines[0])
x1 = list(map(int, lines[1].split()))
y1 = list(map(int, lines[2].split()))
x2 = list(map(int, lines[3].split()))
y2 = list(map(int, lines[4].split()))

```

#### 206 反转链表

```python
206. 反转链表
# Definition for singly-linked list.
 class ListNode:
     def __init__(self, x):
         self.val = x
         self.next = None
        
 class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        # 多元赋值的时候，右边的值不会随着赋值而改变 
        # p = head
        # prev = None
        # while p:
        #     prev, prev.next, p = p, prev, p.next
        # return prev
        
        curr = head
        prev = None
        while curr:
            temp = curr.next
            curr.next = prev
            prev = curr
            curr = temp
        return prev
   		 #递归
    	#if head == None or head.next == None: 
        #     return head
        # p = self.reverseList(head.next)
        # head.next.next = head
        # head.next = None
        # return p
#递归实现用到了python的引用传递性质： Python参数传递统一使用的是引用传递方式。因为Python对象分为可变对象(list,dict,set等)和不可变对象(number,string,tuple等)，当传递的参数是可变对象的引用时，因为可变对象的值可以修改，因此可以通过修改参数值而修改原对象，这类似于C语言中的引用传递；当传递的参数是不可变对象的引用时，虽然传递的是引用，参数变量和原变量都指向同一内存地址，但是不可变对象无法修改，所以参数的重新赋值不会影响原对象，这类似于C语言中的值传递。


```

#### 21 合并两个有序链表

```python
21. 合并两个有序链表
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#    
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        
        # if l1 is None:
        #     return l2
        # elif l2 is None:
        #     return l1
        # elif l1.val < l2.val:
        #     l1.next = self.mergeTwoLists(l1.next, l2)
        #     return l1
        # else:
        #     l2.next = self.mergeTwoLists(l1, l2.next)
        #     return l2
        
        prehead = ListNode(-1)
        prev = prehead
        while l1 and l2:
            if l1.val < l2.val:
                prev.next = l1
                l1 = l1.next
            else:
                prev.next = l2
                l2 = l2.next
            prev = prev.next
        prev.next = l1 if l1 else l2
        return prehead.next
```

#### 23. 合并K个排序链表

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        if not lists:return 
        n = len(lists)
        return self.merge(lists, 0, n-1)
    
    def merge(self, lists, i, j):
        if i == j:
            return lists[i]
        mid = i + (j - i) // 2
        l1 = self.merge(lists, i, mid)
        l2 = self.merge(lists, mid+1, j)
        return self.mergeTwoLists(l1, l2)
            
    def mergeTwoLists(self, l1, l2):
        if not l1:
            return l2
        if not l2:
            return l1
        if l1.val < l2.val:
            l1.next = self.mergeTwoLists(l1.next, l2)
            return l1
        else:
            l2.next = self.mergeTwoLists(l1, l2.next)
            return l2
```



#### 234 回文链表

```python
234. 回文链表
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def isPalindrome(self, head: ListNode) -> bool:
        
        value = []
        while head:
            value.append(head.val)
            head = head.next
         
        return value[::-1] == value
        
#         fast, slow = head, head
#         while fast and fast.next:
#             fast = fast.next.next
#             slow = slow.next
            
#         prev = None
#         curr = slow
#         while curr:
#             nextTemp = curr.next
#             curr.next = prev
#             prev = curr
#             curr = nextTemp
            
#         while prev and head:
#             if prev.val != head.val:
#                 return False
#             prev = prev.next
#             head = head.next
#         return True
```



#### 141 环形链表

```python
141. 环形链表
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        
        if head is None or head.next is None:
            return False
        slow, fast  = head, head.next
        while slow != fast:
            if fast is None or fast.next is None:
                return False
            slow = slow.next
            fast = fast.next.next
        return True
        
        # if head is None or head.next is None:
        #     return False
        # target = {head}
        # head = head.next
        # while head:
        #     if head in target:
        #         print(head.val)
        #         return True
        #     else:
        #         target.add(head)
        #     head = head.next
        # return False
```

#### 19.删除链表的倒数第N个节点

```python
19. 删除链表的倒数第N个节点
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def removeNthFromEnd(self, head, n):
        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """
        
        dummy = ListNode(-1)
        dummy.next = head
        first, second = dummy, dummy
        for i in range(n+1):
            first = first.next
            
        while first:
            first = first.next
            second = second.next
            
        second.next = second.next.next
        return dummy.next
```

#### 20. 有效的括号

```python
class Solution(object):
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        # if len(s) % 2 != 0:
        #     return False
        mappings = {')':'(', ']':'[', '}':'{'}
        stack = []
        for char in s:
            if char in mappings:
                top_element = stack.pop() if stack else '#'
                if top_element != mappings[char]:
                    return False
                
            else:
                stack.append(char)
                
        return not stack
```

#### 32. 最长有效括号

```python
class Solution:
    def longestValidParentheses(self, s: str) -> int:
        stack = []
        stack.append(-1)
        max_len = 0
        for i, char in enumerate(s):
            if char == '(':
                stack.append(i)
            elif char == ')':
                stack.pop()
                if not stack:
                    stack.append(i) # enter current index if stack is empty
                elif stack:
                    max_len = max(max_len, i - stack[-1])
                
        return max_len
        
#         max_len = 0
#         left, right = 0, 0
        
#         for char in s:
#             if char == '(':
#                 left += 1
#             if char == ')':
#                 right += 1
#             if right == left:
#                 max_len = max(max_len, 2*left)
#             if right > left:
#                 left, right = 0, 0
        
#         left, right = 0, 0                
#         for char in reversed(s):
#             if char == '(':
#                 left += 1
#             if char == ')':
#                 right += 1
#             if left == right:
#                 max_len = max(max_len, 2*left)
#             if left > right:
#                 left, right = 0, 0
                
#         return max_len
```



#### 155. 最小栈

```python
class MinStack(object):

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack = []
        self.min_stack = []

    def push(self, x):
        """
        :type x: int
        :rtype: None
        """
        self.stack.append(x)
        if not self.min_stack or x <= self.min_stack[-1]:
            self.min_stack.append(x)

    def pop(self):
        """
        :rtype: None
        """
        if self.stack.pop() == self.min_stack[-1]:
            self.min_stack.pop()

    def top(self):
        """
        :rtype: int
        """
        return self.stack[-1]

    def getMin(self):
        """
        :rtype: int
        """
        return self.min_stack[-1]
```



#### 232. 用栈实现队列

```python
class MyQueue(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.stack_in = []
        self.stack_out = []

    def push(self, x):
        """
        Push element x to the back of queue.
        :type x: int
        :rtype: None
        """
        self.stack_in.append(x)

    def pop(self):
        """
        Removes the element from in front of queue and returns that element.
        :rtype: int
        """
        if not self.stack_out:
            while self.stack_in:
                self.stack_out.append(self.stack_in.pop())
        return self.stack_out.pop()

    def peek(self):
        """
        Get the front element.
        :rtype: int
        """
        if self.stack_out:
            return self.stack_out[-1]
        else:
            return self.stack_in[0]

    def empty(self):
        """
        Returns whether the queue is empty.
        :rtype: bool
        """
        
        if self.stack_in or self.stack_out:
            return False
        else:
            return True
```

#### 496. 下一个更大元素 I

```python
class Solution(object):
    def nextGreaterElement(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
#         ans = [-1] * len(nums1)
        
#         for i, num1 in enumerate(nums1):
#             for num2 in nums2[nums2.index(num1):]:
#                 if num2 > num1:
#                     ans[i] = num2
#                     break
                    
#         return ans
        
        ans = [-1] * len(nums1)
        d = {}
        for i, num in enumerate(nums1):
            d[num] = i
            
        stack = []
        for n in nums2:
            while stack and stack[-1] < n:
                top = stack.pop()
                if top in d:
                    ans[d[top]] = n
            stack.append(n)
            
        return ans
```

#### 150. 逆波兰表达式求值

```python
class Solution:
    def evalRPN(self, tokens: List[str]) -> int:
        
        stack = []
        signs = ['+', '-', '*', '/']
        for c in tokens:
            if not c in signs:
                stack.append(int(c))
            else:
                right = stack.pop()
                left = stack.pop()
                if c == '+':
                    res = left + right
                elif c == '-':
                    res = left - right
                elif c == '*':
                    res = left * right
                elif c== '/':
                    res = int(left / right)
                
                stack.append(res)
            
        return stack[0]
```



#### 224. 基本计算器

```python
class Solution(object):
    def calculate(self, s):
        """
        :type s: str
        :rtype: int
        """
        res, num, sign = 0, 0, 1
        stack = []
        for c in s:
            if c.isdigit():
                num = 10 * num + int(c)
            elif c == "+" or c == "-":
                res = res + sign * num
                num = 0
                sign = 1 if c == "+" else -1
            elif c == "(":
                stack.append(res)
                stack.append(sign)
                res = 0
                sign = 1
            elif c == ")":
                res = res + sign * num
                num = 0
                res *= stack.pop()
                res += stack.pop()
        res = res + sign * num
        return res
```

#### 938. 二叉搜索树的范围和

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def rangeSumBST(self, root: TreeNode, L: int, R: int) -> int:
        
        # calculate the sum of value that L <= value <= R
        
        if root == None:
            return 0
        elif root.val < L:
            return self.rangeSumBST(root.right, L, R)
        elif root.val > R:
            return self.rangeSumBST(root.left, L, R)
        else:
            return root.val + self.rangeSumBST(root.left, L, R) + self.rangeSumBST(root.right, L, R)
```

#### 783. 二叉搜索树结点最小距离

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def minDiffInBST(self, root: TreeNode) -> int:
        
        def preOrder(root):
            if root is None:
                return []
            
            return preOrder(root.left) + [root.val] + preOrder(root.right)
        
        root_val = preOrder(root)
        min_value = 1000
        for i in range(len(root_val)-1):
            min_value = min(root_val[i+1] - root_val[i], min_value)
            
        return min_value
```

#### 94. 二叉树的中序遍历

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        
        stack = []
        res = []
        curr = root
        
        while curr or stack:
            while curr:
                stack.append(curr)
                curr = curr.left
                
            curr = stack.pop()
            res.append(curr.val)
            curr = curr.right
        
        return res
        
        
#         if root is None:
#             return []
        
#         return self.inorderTraversal(root.left) + [root.val] + self.inorderTraversal(root.right)
        
```

#### 56. 合并区间

```python
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        
        intervals = sorted(intervals, key=lambda x: x[0])
        
        merged = []
        if intervals:
            merged.append(intervals[0])
        
        for interval in intervals:
            if merged[-1][1] < interval[0]:
                merged.append(interval)
            elif merged[-1][1] >= interval[0]:
                merged[-1][1] = max(merged[-1][1], interval[1])
                
        return merged
        
```

#### Merge Sort

```python
def merge_sort(a):
    merge_sort_c(a, 0, len(a)-1)

def merge_sort_c(a, p, r):

    if p < r:
        mid = (r - p) // 2 + p
        merge_sort_c(a, p, mid)
        merge_sort_c(a, mid+1, r)
        merge(a, p, mid, r) 

def merge(a, low, mid, high):
    temp = []
    p = low
    q = mid+1
    while p <= mid and q <= high:
        if a[p] <= a[q]:
            temp.append(a[p])
            p += 1
        else:
            temp.append(a[q])
            q += 1

    start = p if p <= mid else q
    end = mid if p <= mid else high
    temp.extend(a[start:end+1])
    a[low:high + 1] = temp
```

#### Quick Sort

```python
# test
# n = [1, 2, 4, 5, 3, 1]
# quickSort(n)
# print(n)
def quick_sort(a):
    quick_sort_c(a, 0, len(a)-1)

def quick_sort_c(a, low, high):

    if low < high:
        q = partition(a, low, high)
        quick_sort_c(a, low, q-1)
        quick_sort_c(a, q+1, high)

def partition(a, low, high):
    piviot = a[high]
    i, j = low, low
    while j <= high:
        if a[j] < piviot:
            a[i], a[j] = a[j], a[i]
            i += 1
        j += 1

    a[i], a[high] = a[high], a[i]
    return i
```

#### 230. 二叉搜索树中第K小的元素

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def kthSmallest(self, root: TreeNode, k: int) -> int:
        
        # generator, early stop
        def mid_order(root):
            if root is None: return
            yield from mid_order(root.left)
            yield root.val
            yield from mid_order(root.right)
            
        gen = mid_order(root)
        for i in range(k-1):
            next(gen)
        return next(gen)
        
        
        # inorder traversal
#         def mid_order(root):
            
#             res = []
#             if root is None:
#                 return []
#             res = mid_order(root.left) + [root.val] + mid_order(root.right)
#             return res
        
#         res = []
#         res = mid_order(root)
#         return res[k-1]
```

#### 146. LRU缓存机制

```python
class LRUCache:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache_list = []
        self.cache = {}

    def get(self, key: int) -> int:
        if key in self.cache_list:
            self.cache_list.remove(key)
            self.cache_list.append(key)
            return self.cache[key]
        else:
            return -1

   def put(self, key: int, value: int) -> None:
        
        if key in self.cache_list:
                self.cache_list.remove(key)
                self.cache_list.append(key)
                self.cache[key] = value
        else:
            if len(self.cache_list) == self.capacity:
                del self.cache[self.cache_list[0]]
                self.cache_list = self.cache_list[1:]

            self.cache_list.append(key)
            self.cache[key] = value
        
        


# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)
```

#### 895. 最大频率栈

```python
from collections import Counter, defaultdict
class FreqStack:

    def __init__(self):
        
        self.freq = Counter()
        self.group = defaultdict(list)
        self.max_freq = 0
        

    def push(self, x: int) -> None:
        
        f = self.freq[x] + 1
        self.freq[x] = f
        if f > self.max_freq:
            self.max_freq = f
        self.group[f].append(x)
       

    def pop(self) -> int:
        
        x = self.group[self.max_freq].pop()
        self.freq[x] -= 1
        if not self.group[self.max_freq]:
            self.max_freq -= 1
            
        return x

# Your FreqStack object will be instantiated and called as such:
# obj = FreqStack()
# obj.push(x)
# param_2 = obj.pop()
```

#### 15. 三数之和

```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        
        res = []
        nums.sort()
        for i in range(len(nums)-2):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            target = 0 - nums[i]
            start, end = i + 1, len(nums) - 1
            while start < end:
                if nums[start] + nums[end] > target:
                    end -= 1  
                elif nums[start] + nums[end] < target:
                    start += 1
                else:
                    res.append((nums[i], nums[start], nums[end]))
                    end -= 1
                    start += 1
                    while start < end and nums[end] == nums[end + 1]:
                        end -= 1
                    while start < end and nums[start] == nums[start - 1]:
                        start += 1
        return res
```

#### 70. 爬楼梯

```python
class Solution:
    def climbStairs(self, n: int) -> int:
        
        
        # fibonacci
        # a = 1
        # b = 1
        # for i in range(n-1):
        #     a, b = a+b, b
        # return a
    
        if n == 1:
            return 1
        dp = {}
        dp[1] = 1
        dp[2] = 2
        for i in range(3, n+1):
            dp[i] = dp[i-1] + dp[i-2]
            
        return dp[n]
        
```

#### 239. 滑动窗口最大值

```python
from typing import List
from collections import deque


class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        size = len(nums)

        # 特判
        if size == 0:
            return []
        # 结果集
        res = []
        # 滑动窗口，注意：保存的是索引值
        window = deque()

        for i in range(size):
            # 当元素从左边界滑出的时候，如果它恰恰好是滑动窗口的最大值
            # 那么将它弹出
            if i >= k and i - k == window[0]:
                window.popleft()

            # 如果滑动窗口非空，新进来的数比队列里已经存在的数还要大
            # 则说明已经存在数一定不会是滑动窗口的最大值（它们毫无出头之日）
            # 将它们弹出
            while window and nums[window[-1]] <= nums[i]:
                window.pop()
            window.append(i)

            # 队首一定是滑动窗口的最大值的索引
            if i >= k - 1:
                res.append(nums[window[0]])
        return res
```

#### 69. x 的平方根

```python
class Solution:
    def mySqrt(self, x: int) -> int:
        
        res = 1
        while abs(res**2 - x) > 1e-3:
            res = (res + x / res) / 2
            
        return int(res)
        
#         if x == 0 or x == 1:
#             return x
        
#         l, r = 1, x
        
#         while l <= r:
#             m = l + (r - l) // 2
#             if m **2 <= x < (m + 1) ** 2:
#                 return m
#             elif m ** 2 < x:
#                 l = m + 1
#             else:
#                 r = m - 1
    
```

#### 33. 搜索旋转排序数组

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        
#         if target in nums:
#             return nums.index(target)
#         else:
#             return -1
        
        
        return self.helper(nums,0,len(nums)-1,target)
    
    def helper(self,nums,low,high,target):
        if low>high:
            return -1
        mid = (low+high)//2
        if nums[mid] == target:
            return mid
        
        # 如果右左半部分是有序数列
        if nums[mid]<nums[high]:
            if nums[mid] < target and target <= nums[high]: 
                return self.helper(nums,mid+1,high,target)
            else:
                return self.helper(nums,low,mid-1,target)  

        else:
            if nums[low] <= target and target < nums[mid]:  
                return self.helper(nums,low,mid-1,target)
            else:
                return self.helper(nums,mid+1,high,target)  
       
```

#### 64. 最小路径和

```python
import numpy as np
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        
        m, n = len(grid), len(grid[0])
        
        for i in reversed(range(m)):
            for j in reversed(range(n)):
                if i == m-1 and j != n-1:
                    grid[i][j] = grid[i][j] + grid[i][j+1]
                elif i != m-1 and j == n-1:
                    grid[i][j] = grid[i][j] + grid[i+1][j]
                elif i != m-1 and j!= n-1:
                    grid[i][j] = grid[i][j] + min(grid[i+1][j], grid[i][j+1])
        
        return grid[0][0]
        
#         m, n = len(grid), len(grid[0])
#         path_sum = [[np.inf] * n] * m
#         path_sum[0][0] = grid[0][0]
        
#         for i in range(m):
#             for j in range(n):
#                 if i > 0 and j > 0:
#                     path_sum[i][j] = grid[i][j] + min(path_sum[i-1][j], path_sum[i][j-1])
#                 elif j == 0 and i > 0:
#                     path_sum[i][j] = grid[i][j] + path_sum[i-1][j]
#                 elif i == 0 and j > 0:
#                     path_sum[i][j] = grid[i][j] + path_sum[i][j-1]
                    
#         return path_sum[m-1][n-1]
```

#### 322. 零钱兑换

```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        
        ans = [0 for _ in range(amount+1)]
        
        for i in range(1, amount+1):
            cost = float('inf')
            for c in coins:
                if i - c >=0:
                    cost = min(cost, ans[i-c] + 1)
                    
            ans[i] = cost
                
        return ans[amount] if ans[amount] != float('inf') else -1
```

#### 121. 买卖股票的最佳时机

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        
        max_p = 0
        if not prices:
            return 0
        min_p = prices[0]
        
        for i in range(1, len(prices)):
            min_p = min(min_p, prices[i])
            max_p = max(max_p, prices[i] - min_p)
            
        return max_p
        
```

