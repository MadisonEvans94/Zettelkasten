#seed 
upstream:

---

**video links**: 

---

# Brain Dump: 
- tortoise and the hare algorithm and when to use for linked lists

```python
slow = fast = head

while fast.next and fast.next.next:

	slow = slow.next
	fast = fast.next.next
	
	# slow is now the middle of the linked list
	prev, curr = None, slow.next
```

- BKM for detecting loops in LL 
```python 
class ListNode(object):

def __init__(self, x):

self.val = x

self.next = None

  

class Solution(object):

def hasCycle(self, head):

"""

:type head: ListNode

:rtype: bool

"""

# Initialize fast and slow pointers

if not head or not head.next:

return False

  

slow_node, fast_node = head, head.next

  

# Single pass

while fast_node and fast_node.next:

if slow_node == fast_node:

return True

  

slow_node = slow_node.next

fast_node = fast_node.next.next

  

return False
```

--- 








This document aims to provide a comprehensive guide on how to approach LeetCode problems based on specific keywords and patterns. For each keyword or pattern, a short Python code snippet is provided to demonstrate useful methods and coding approaches.

---

## Table of Contents

1. [Arrays and Strings](#arrays-and-strings)
2. [Linked Lists](#linked-lists)
3. [Stacks and Queues](#stacks-and-queues)
4. [Hash Tables](#hash-tables)
5. [Trees and Graphs](#trees-and-graphs)
6. [Heaps](#heaps)
7. [Dynamic Programming](#dynamic-programming)
8. [Sets](#sets)
9. [Bit Manipulation](#bit-manipulation)

---

## Arrays and Strings

### Contiguous Subarray

**Keywords**: Contiguous, Subarray

**Strategy**: Use arrays or strings directly, sometimes with a two-pointer technique.

```python
# Two-pointer technique to find a target sum in a sorted array
left, right = 0, len(arr) - 1
while left < right:
    current_sum = arr[left] + arr[right]
    if current_sum == target:
        return True
    elif current_sum < target:
        left += 1
    else:
        right -= 1
```

### Sorting

**Keywords**: Sorting, Order

**Strategy**: Use built-in sorting methods for arrays or custom sorting methods for strings based on the problem requirements.

#### Arrays

For arrays, you can use Python's built-in `sort()` method, which sorts the array in-place.

```python
# Using Python's built-in sort for arrays
arr.sort()
```

#### Strings

Strings in Python are immutable, so you can't sort them in-place like arrays. If you need a sorted string, you can use the `sorted()` function, which returns a list of sorted characters, and then join them back into a string.

```python
# Sorting and joining for strings
sorted_word = ''.join(sorted(word))
```

**When to Use Which?**

- Use `arr.sort()` when you're working with a mutable sequence like a list and you want to sort it in-place.
- Use `''.join(sorted(word))` when you're working with strings or when you need a new sorted string without altering the original string.

---


Certainly! Let's expand the Linked Lists section to include the creation of a linked list and the essential methods it should have.

---

## Linked Lists

### Basic Structure

Here's how you can define a simple node for a singly linked list in Python:

```python
# Node class for singly linked list
class Node:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
```

### Essential Methods

#### Append

To add an element to the end of the linked list.

```python
def append(self, val):
    new_node = Node(val)
    if not self.head:
        self.head = new_node
        return
    last_node = self.head
    while last_node.next:
        last_node = last_node.next
    last_node.next = new_node
```

#### Prepend

To add an element to the beginning of the linked list.

```python
def prepend(self, val):
    new_node = Node(val)
    new_node.next = self.head
    self.head = new_node
```

#### Delete Node

To delete a node by value.

```python
def delete_node(self, key):
    current_node = self.head
    if current_node and current_node.val == key:
        self.head = current_node.next
        current_node = None
        return
    prev = None
    while current_node and current_node.val != key:
        prev = current_node
        current_node = current_node.next
    if current_node:
        prev.next = current_node.next
        current_node = None
```

### Use-Cases

#### Sequential Access

**Keywords**: Sequential Access

**Strategy**: Use linked lists for sequential data access.

```python
# Traversing a linked list
current = head
while current:
    print(current.val)
    current = current.next
```
Certainly! I've updated the "Cycle Detection" subsection to include a high-level explanation of Floyd's cycle detection algorithm, how it works, and its utility.

---

#### Cycle Detection

**Keywords**: Cycle, Loop, Circular

**Strategy**: Use Floyd's cycle detection algorithm, also known as the "tortoise and hare" algorithm.

**Explanation**: Floyd's algorithm uses two pointers that move through the list at different speeds. The slow pointer moves one step at a time while the fast pointer moves two steps. If there is a loop, the fast pointer will eventually meet the slow pointer within that loop. If there's no loop, the fast pointer will reach the end of the list.

This algorithm is used because it detects cycles in O(n) time complexity and uses O(1) extra space, making it highly efficient.

```python
# Floyd's cycle detection
slow, fast = head, head
while fast and fast.next:
    slow = slow.next
    fast = fast.next.next
    if slow == fast:
        return True
```

---

## Stacks and Queues

### Last-In, First-Out (LIFO)

**Keywords**: Reversing, Backtracking, LIFO

**Strategy**: Use a stack for "last-in, first-out" behavior.

```python
# Using a stack to reverse a string
stack = []
for char in string:
    stack.append(char)
reversed_string = ''.join(stack.pop() for _ in range(len(stack)))
```

### First-In, First-Out (FIFO)

**Keywords**: FIFO, Level-order

**Strategy**: Use a queue for "first-in, first-out" behavior or level-order traversal.

```python
from collections import deque

# Using a queue for level-order traversal in a binary tree
queue = deque([root])
while queue:
    node = queue.popleft()
    if node:
        print(node.val)
        queue.append(node.left)
        queue.append(node.right)
```

---

Certainly! I've updated the "Hash Tables" section to focus on Python dictionaries, which are the Pythonic way to implement hash tables.

---

## Hash Tables (Dictionaries in Python)

### Lookup

**Keywords**: Lookup, Constant-time, Key-Value Pair

**Strategy**: Use Python dictionaries for quick lookups and constant-time insertions. Dictionaries are essentially hash tables in Python.

```python
# Using a Python dictionary for quick lookups
lookup = {key: value for key, value in enumerate(arr)}
```

### Counting

**Keywords**: Counting, Occurrences, Frequency

**Strategy**: Use Python dictionaries to count occurrences of elements. This is a more Pythonic way compared to using specialized collections like `Counter`.

```python
# Counting occurrences of elements in an array using a dictionary
count = {}
for elem in arr:
    if elem in count:
        count[elem] += 1
    else:
        count[elem] = 1
```

---

## Sets

### Uniqueness

**Keywords**: Uniqueness, Duplicates

**Strategy**: Use sets for checking uniqueness or duplicates.

```python
# Using a set to check for duplicates
unique_elements = set(arr)
```

---

## Trees and Graphs

### Hierarchical

**Keywords**: Hierarchical, Ancestors, Descendants

**Strategy**: Use tree structures for hierarchical data.

```python
# Basic binary tree node
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
```

### Connected Components

**Keywords**: Networks, Cities, Relationships

**Strategy**: Use graph structures for connected components.

```python
# Basic graph adjacency list
graph = {node: [neighbors] for node, neighbors in enumerate(arr)}
```

---

## Heaps

### Kth Element

**Keywords**: Kth largest, Kth smallest

**Strategy**: Use a heap to find the Kth element efficiently.

```python
import heapq

# Finding the Kth smallest element
heapq.heapify(arr)
kth_smallest = heapq.nsmallest(k, arr)[-1]
```

---

## Dynamic Programming

### Optimization

**Keywords**: Maximum, Minimum, Most efficient

**Strategy**: Use dynamic programming for optimization problems.

```python
# Basic dynamic programming for Fibonacci sequence
dp = [0, 1]
for i in range(2, n):
    dp.append(dp[i-1] + dp[i-2])
```

---

## Bit Manipulation

### Binary

**Keywords**: Binary, Bitwise

**Strategy**: Use bit manipulation for binary number problems.

```python
# Checking if the ith bit is set
if n & (1 << i):
    print("Bit is set")
```
