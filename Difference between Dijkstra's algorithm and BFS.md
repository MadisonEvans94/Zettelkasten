#seed 
###### upstream: 

### Origin of Thought:


### Underlying Question: 


### Solution/Reasoning: 


### Examples (if any): 

[[Weighted vs Unweighted Graphs]]

Dijkstra's algorithm and Breadth-First Search (BFS) are both algorithms that traverse or search through a graph, but they have different goals and behave differently.

BFS is used for traversing or searching a graph in a breadthward motion. It uses a queue data structure to achieve this. BFS visits nodes level by level in the graph. Starting from a given source node, BFS first visits all nodes at one hop away (direct neighbors), then it visits all nodes at two hops away, and so forth. This makes BFS an excellent tool when you want to find the shortest path in an unweighted graph - that is, a graph where all edges have the same weight or cost.

On the other hand, Dijkstra's algorithm is a bit more sophisticated. While it also searches a graph, its goal is to find the shortest path between two nodes in a graph with weighted edges (where edges have different costs). In Dijkstra's algorithm, the "shortest path" is determined by the sum of the weights of the edges. Unlike BFS, Dijkstra's algorithm takes into account the weights of the edges while determining the shortest path. It uses a priority queue (often implemented as a min heap) as its main data structure.

In summary, Dijkstra's algorithm is essentially a generalization of BFS that can handle weighted graphs. The key difference is how they deal with weights or costs: BFS treats all edges as equal, while Dijkstra's algorithm takes into account the actual weights of the edges.