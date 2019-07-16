[TOC]



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

