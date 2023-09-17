#evergreen1 
###### upstream: [[Algorithms]], [[Artificial Intelligence]]
###### siblings: 

## TLDR: 
- Greedy algorithms lead to local optima because we are looking for the immediate best 
- It's almost more accurate to call them hungry algorithms; a greedy algorithm will take the best option available to it at each decision point without considering future consequences, just like a hungry dog might eat food put in front of it without considering whether it's the best choice for its long-term health or satisfaction

### Origin of Thought:
- Question emerged while reviewing AI, graph algorithms, and search

### Underlying Question: 
What does it mean when we describe an algorithm as 'greedy', and how does this affect its behavior and performance?

### Solution/Reasoning: 
-   A greedy algorithm is an algorithmic paradigm that follows the problem-solving heuristic of making the locally optimal choice at each stage with the hope that these local solutions will lead to a global optimum.
-   The term 'greedy' comes from the fact that these algorithms make the most optimal (or 'greedy') choice at each decision point. They look for the best solution in the current situation without worrying about the consequences of that decision for future decisions. This can make them efficient in terms of time and computational resources, as they don't need to consider multiple future scenarios before making a decision.
-   However, this approach doesn't always lead to the best overall solution, because making the best decision in the short term doesn't always lead to the best decision in the long term. This is known as the problem of local optima: a solution might be the best within a small range, but not the best overall.


### Examples (if any): 

1.  **[[Dijkstra's Algorithm]] for Shortest Path**: This algorithm constructs the shortest path from a source to all vertices in the graph. At each step, it 'greedily' chooses the closest vertex that has not been processed yet.
    
2.  **Prim's and Kruskal's Algorithms for [[Minimum Spanning Tree]]**: These algorithms add the next smallest weight edge to the tree being constructed, making the 'greedy' choice at each step.
    
3.  **[[Huffman Coding]]**: This is a greedy algorithm used for lossless data compression. It 'greedily' constructs variable length prefix codes based on the frequency of characters in the input data. Characters that occur more frequently are assigned shorter codes, while characters that occur less frequently are assigned longer codes.
    
4.  **[[Activity Selection Problem]]**: Here, the goal is to do the maximum number of activities given start and end times of activities. The greedy choice is to always pick the next activity whose finish time is least among the remaining activities and the start time is more than or equal to the finish time of the previously selected activity.

The opposite of a greedy algorithm could be considered to be a globally optimal algorithm or exhaustive search algorithm. These algorithms don't settle for local optima like greedy algorithms do, but instead explore all possible solutions to find the globally optimal solution.

Here are a couple of examples:

1.  **Brute Force Algorithms**: These algorithms explore all possible solutions to a problem and pick the best one. They guarantee to find a global optimum, but they are often inefficient because they don't make any attempt to prune the search space.
    
2.  **Dynamic Programming Algorithms**: These algorithms solve complex problems by breaking them down into simpler subproblems and solving each of these subproblems just once, storing their results in case the same subproblem is encountered again. This is unlike greedy algorithms, which make an irrevocable choice at each step. Instead, dynamic programming combines the solutions to the subproblems to reach the final solution, ensuring a global optimum.
    
3.  **Backtracking Algorithms**: These algorithms explore all possible solutions to a problem by incrementally building candidates to the solutions, and abandoning a candidate as soon as it is determined that the candidate cannot possibly be extended to a valid solution. This is also unlike greedy algorithms, which make a single choice and stick with it.
    

In general, these types of algorithms can find the global optimum, but they can be more time and space-consuming than greedy algorithms, which are more efficient but may only find a local optimum.