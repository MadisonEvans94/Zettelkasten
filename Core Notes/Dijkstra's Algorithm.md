

#incubator 
upstream: [[Algorithms]], [[Graph Theory]]

---

**links**:
[Computerphile Video](https://www.youtube.com/watch?v=GazC3A4OQTE&t=547s&ab_channel=Computerphile)
[Python Implementation](https://www.youtube.com/watch?v=XEb7_z5dG3c&pp=ygURbmVldGNvZGUgZGlqa3N0cmE%3D)

---

## Overview of Dijkstra's Algorithm

### What is Dijkstra's Algorithm?

Dijkstra's Algorithm is a fundamental algorithm in computer science for finding the shortest paths from a single source node to all other nodes within a graph. This graph can be either directed or undirected, and it should have non-negative weights on its edges. The algorithm efficiently computes the shortest path distances and can also be modified to output the paths themselves.

### Why is it Used?

The primary use of Dijkstra's Algorithm is in solving the single-source shortest path problem, where the goal is to find the minimum distance from a given source node to every other node in the graph. Its importance lies in its efficiency and accuracy for graphs with non-negative weights, making it a crucial tool for routing and navigation tasks. Dijkstra's Algorithm is favored for its straightforward implementation and its ability to handle dynamically changing graphs, which is essential for real-time applications.

### Where is it Commonly Used?

Dijkstra's Algorithm finds applications in a wide array of fields, including but not limited to:

- **Network Routing:** Used in protocols like OSPF (Open Shortest Path First) and IS-IS (Intermediate System to Intermediate System) for calculating the best routing paths in a computer network.
- **Geographical Mapping:** Essential for GPS and mapping services to find the shortest routes between physical locations.
- **Telecommunications:** Utilized in the management of telecommunications networks, where finding the least cost path through a network is crucial for efficient data transmission.
- **Robotics:** Applied in path planning for robots to navigate efficiently in an environment.
- **Video Games:** Used in game development for pathfinding, helping characters navigate through the game world.

Dijkstra's Algorithm is celebrated for its versatility and has become a cornerstone in the field of computer science and operations research, serving as a critical tool in the design and analysis of network systems across various domains.

---
## Python Implementation 

```python
import heapq
from typing import List, Dict, Tuple

def shortestPath(self, n: int, edges: List[List[int]], src: int) -> Tuple[Dict[int, int], Dict[int, List[int]]]:
    
    adj = {i: [] for i in range(n)}
    for source, destination, weight in edges:
        adj[source].append((weight, destination))

    shortest = {}
    predecessor = {i: None for i in range(n)}
    minHeap = []
    heapq.heappush(minHeap, (0, src))

    while minHeap:
        w1, n1 = heapq.heappop(minHeap)
        if n1 in shortest:
            continue
        shortest[n1] = w1
        for w2, n2 in adj[n1]:
            if n2 not in shortest:
                heapq.heappush(minHeap, (w1 + w2, n2))
                # Update the predecessor if this is a newly discovered shorter path
                if n2 not in shortest or w1 + w2 < shortest[n2]:
                    predecessor[n2] = n1

    paths = {}
    for target in range(n):
        if target in shortest:
            path = []
            while target is not None:
                path.append(target)
                target = predecessor[target]
            paths[target] = path[::-1]  # Reverse the path to start from the source
        else:
            paths[target] = []

    # Fix to include -1 for unreachable nodes in shortest distances
    for i in range(n):
        if i not in shortest:
            shortest[i] = -1

    return shortest, paths

# Example Implementation 

edges = [
    [0, 1, 4],
    [0, 2, 1],
    [2, 1, 2],
    [1, 3, 1],
    [2, 3, 5],
    [3, 4, 3]
]

shortestPath(n=5, edges=edges, src=0)

# ------------------------------

# Output 
({
    0: 0,  # Distance to itself
    1: 3,  # Shortest distance to node 1
    2: 1,  # Shortest distance to node 2
    3: 4,  # Shortest distance to node 3
    4: 7   # Shortest distance to node 4
}, {
    0: [0],        # Path to itself
    1: [0, 2, 1],  # Path to node 1 through node 2
    2: [0, 2],     # Path to node 2 directly
    3: [0, 2, 1, 3], # Path to node 3 through nodes 2 and 1
    4: [0, 2, 1, 3, 4] # Path to node 4 through nodes 2, 1, and 3
})


```

---
## Application: Linkstate Routing 

In **Link-State Routing** protocols that use Dijkstra's algorithm, the primary goal is to compute the shortest path lengths from a source node to all other nodes in the network. This computation helps in routing decisions by identifying the most efficient paths for packet forwarding. However, knowing the shortest path lengths alone isn't enough for actual packet forwarding in a network; routers also need to know the specific sequence of nodes (or routers) to traverse to reach each destination according to these shortest paths. This requirement leads to the need for path reconstruction.

The basic version of Dijkstra's algorithm focuses on calculating the shortest distances, but with a slight modification, it can also track the actual paths. This is typically achieved by maintaining a predecessor or **parent node** for each node in the network. As the algorithm progresses and updates the shortest distances to each node, it also updates the predecessor of each node to reflect the path taken to achieve that shortest distance.

Once the algorithm completes, you can reconstruct the shortest path from the source to any destination by following these predecessor links backward from the destination node to the source node. This process involves starting at the destination node, looking up its predecessor, then looking up the predecessor's predecessor, and so on, until reaching the source node. The sequence of nodes encountered during this backtracking forms the actual path taken.

Implementing path reconstruction in Dijkstra's algorithm, as shown in the modified version of your function, allows routing protocols to not only determine the optimal distances but also to utilize the specific paths for packet forwarding. This capability is crucial for the practical application of these algorithms in network routing, ensuring that data packets are sent through the most efficient routes as determined by the algorithm.

In summary, while the basic version of Dijkstra's algorithm calculates shortest path lengths, an enhanced version that includes path reconstruction is essential for actual network routing operations. This enhancement enables routers to make informed forwarding decisions, routing packets over the network through the optimal paths as identified by the algorithm.