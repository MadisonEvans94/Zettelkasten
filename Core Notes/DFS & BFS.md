#incubator 
###### upstream: [[Algorithms]]

[[DFS & BFS Example Implementation ]]

### Origin of Thought:
- these are some of the most basic building blocks when it comes to search algorithms 


### Underlying Question: 
- How are BFS & DFS similar and different? 
- How do you implement them from scratch, and what basic data structures do you need to do so efficiently? 
- Where do DFS and BFS typically appear the most? 
- How will I use these algorithms as a full stack engineer? ML engineer? Systems? 

### Solution/Reasoning: 
Both algorithms are fundamental strategies for traversing a graph or tree structure


- **BFS** 
	- traverses in a breadthward motion, kinda like how heat is dispersed evenly from a source 
	- it uses a queue data structure to keep track of which node to check next 
- **DFS**
	- traverses by exploring a path all the way before hitting a dead end and backtracking 
	- it uses a stack data structure to keep track of which node to check next 

As a **Full Stack Engineer**, you may encounter problems like traversing or searching data structures, parsing nested structures (like JSON/XML objects), or using tree/graph traversal for UI rendering algorithms or even managing task dependencies.

As a **Machine Learning Engineer**, you may use BFS/DFS in decision trees, navigation through state-space problems, feature extraction from hierarchical structures, etc. For instance, Random Forest algorithm builds multiple decision trees, and BFS/DFS are used to traverse these trees.

As a **Systems Engineer**, you may use these in network troubleshooting, parsing system files, and directories.


### Examples (if any): 
*In these examples, `graph` is a dictionary where the keys are node identifiers and the values are lists of nodes that the key node has a direct path to.*

**BFS** and **DFS**

```python
from collections import deque

def graph_traversal(graph, root, data_structure):
    visited = set()
    data_structure.append(root)

    while data_structure:
        if isinstance(data_structure, deque):
            vertex = data_structure.popleft() # BFS
        else:
            vertex = data_structure.pop() # DFS

        if vertex not in visited:
            visited.add(vertex)
            print(vertex, end=" ")

            for neighbour in graph[vertex]:
                if neighbour not in visited:
                    data_structure.append(neighbour)

```


-   Takes a graph and a starting point as arguments.
-   Initializes an empty set called visited that will keep track of all the nodes that have been visited.
-   Appends the starting point to the provided data structure (`data_structure`), which can be either a queue (for BFS) or a stack (for DFS).
-   Enters a loop that continues while `data_structure` isn't empty.
-   In each iteration of the loop:
    -   If `data_structure` is a queue (`isinstance(data_structure, deque)`), it performs a `popleft` operation to get the next vertex (BFS behavior).
    -   If `data_structure` is a stack (a list in this context), it uses `pop` to get the next vertex (DFS behavior).
    -   If the current vertex has not been visited, it adds the vertex to the visited set and then prints it.
    -   It then iterates over each neighbor of the current vertex (`graph[vertex]`). If a neighbor isn't in `visited`, then it appends the neighbor to `data_structure`.
-   This process continues until all reachable nodes have been visited, ensuring a complete traversal of the graph according to BFS or DFS, depending on the data structure used.
