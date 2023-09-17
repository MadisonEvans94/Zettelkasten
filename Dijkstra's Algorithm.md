
###### Upstream: [[Algorithms]], [[Graph Theory]]
###### Siblings: [[shortest path]], 
#incubator 

2023-05-21
19:19

[Computerphile Video](https://www.youtube.com/watch?v=GazC3A4OQTE&t=547s&ab_channel=Computerphile)
[Python Implementation](https://www.youtube.com/watch?v=OrJ004Wid4o&ab_channel=ThinkXAcademy)

### Main Takeaways
-   Dijkstra's Algorithm is a method for finding the shortest path between two points in a graph. It is like a magical map that always knows the quickest route to any given destination.
-   Just as you would navigate a large amusement park to reach a particular ride as quickly as possible, Dijkstra's Algorithm systematically calculates the shortest path from a starting point (like the Ferris wheel) to all other points (like other rides in the park).

### Why
-   Dijkstra's Algorithm is used when we want to find the shortest path between two points. In our analogy, this is like trying to find the quickest route from the Ferris wheel to the roller coaster in an amusement park.
-   It's useful in many real-world applications, such as GPS navigation systems, network routing protocols, and in games for determining paths of characters.

### How

-   The algorithm starts at the initial point (the Ferris wheel), and initially marks the distance to all other points as infinity (∞) because it doesn't know yet how long it will take to reach them.
-   It then checks all direct paths from the initial point to the nearby points and updates the time it takes to reach them.
-   It then selects the point with the shortest time, considers all direct paths from that point, and again updates the times it takes to reach other points, if the new path is shorter than the previously recorded one.
-   This process is repeated until it has visited every point, always choosing the point with the shortest unvisited path next.
-   At the end, the shortest path from the initial point to all other points has been determined. You just follow the path that the algorithm has laid out to reach your destination!
- it often uses a [[Priority Queue]]

### Example 

```python
function Dijkstra(Graph, source): 
	# create vertex set Q 
	Q = set of all nodes in Graph 
	
	# set distance to zero for our initial node and to infinity for other nodes 
	distance = dict with key as node and value as float('inf') for all nodes in Graph 
	distance[source] = 0 
	
	# set of visited nodes 
	visited = empty set 
	
	# previous node in optimal path from source 
	previous = dict
	
	while Q is not empty: 
		# node with the least distance selected first 
		node = minimum distance node in Q 
		
		# remove the visited node from Q 
		Q.remove(node) 
		
		# add node to visited nodes 
		visited.add(node) for each neighbor of node: 
			alt = distance[node] + length of edge from node to neighbor 
			if alt < distance[neighbor]: 
				distance[neighbor] = alt 
				previous[neighbor] = node
	return distance, previous
```


### Additional Questions: 
- [ ] [[Difference between Dijkstra's algorithm and BFS]]