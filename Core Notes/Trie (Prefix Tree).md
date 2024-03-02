#seed 
upstream: [[Data Structures]]

---

**links**: 

---

## Overview 

- Insert a Word in `O(1)`
- Search a Word in `O(1)`
- Search a Prefix in `O(1)`

> In this context, by **word** we mean a string of characters. Also by constant time, we mean k where k is the length of the word, which relatively speaking is constant 


**Why not just use a hash table?**

Sure a hash table allows us to look up and insert in constant time, however, we cannot search based on prefixes. For example, if we have the prefix `ap` and we want to know which items (if any) start with that prefix, it would be pretty difficult to implement in a hash table, whereas this is the exact use can a prefix tree specializes in 

**Where is it used?**
- search engines 
- networking routing tables 
- nlp applications 

## Implementation 

### Intuition 
Each node is a letter of the alphabet and each node has 26 children. So if you wanted the word apple, you would start at the root (which is `null`) and traverse down through `a` then `p` until you get apple. It is commonly implemented using a `TrieNode` class which has a hash map attribute for storing children and a boolean attribute set to `True` if it is a valid word. 

![[Trie Diagram.png]]

### Python Example 
```python 
class TrieNode: 
	def __init__(self): 
		self.children = {}
		self.word = False

class Trie: 
	def __init__(self): 
		self.root = TrieNode()

	def insert(self, word): 
		curr = self.root
		for c in word: 
			if c not in curr.children: 
				curr.children[c] = TrieNode() 
			curr = curr.children[c]
		curr.word = True 

	def search(self, word): 
		curr = self.root
		for c in word: 
			if c not in curr.children: 
				return False 
			curr = curr.children[c]
		return curr.word

	def startsWith(self, prefix): 
		curr = self.root 
		for c in prefix: 
			if c not in curr.children: 
				return False
			curr = curr.children[c]
		return True 
			
```
