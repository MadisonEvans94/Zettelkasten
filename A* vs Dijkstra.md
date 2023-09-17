#seed 
###### upstream: [[Algorithms]]
###### siblings: [[Dijkstra's Algorithm]], [[A*]]

### Origin of Thought:


### Underlying Question: 


### Solution/Reasoning: 

Let's imagine you're playing a video game, and you're trying to find the quickest way to get from one point to another in the game's world.

**Dijkstra's Algorithm**

Let's start with Dijkstra's algorithm. This is like if you were to try every possible path you could take in the game, starting with the shortest paths and extending them little by little, until you found the quickest path to your destination. You'd keep track of the quickest way to get to each spot you could reach along the way, and you'd use that information to guide your search.

It's like being in a maze and you don't have a map. So, you start exploring all paths. Every time you reach an intersection, you make a decision to continue on the path which is shortest so far. You do this until you finally reach the exit.

__A_ Algorithm_*

On the other hand, A* (A-Star) is a bit smarter. A* also explores paths like Dijkstra's algorithm, but it has a "heuristic," or a way of making a good guess, that helps it prioritize which paths to explore first. It uses this heuristic to guide its search toward the destination more directly, instead of blindly trying all paths.

This is like having a compass in our maze, which always points towards the exit. So, instead of just picking the shortest path at each intersection, now you also consider which path leads you in the direction the compass points.

So, while Dijkstra's algorithm explores all possible paths equally, A* uses some extra knowledge (the heuristic) to explore paths that seem like they're going in the right direction first. This usually lets A* find the quickest path more quickly than Dijkstra's algorithm.

Both of them will find the shortest path eventually, but A* can do it faster if the heuristic is chosen well.

Of course, this is a simplified explanation. There's a lot more math involved in how these algorithms actually work, but this should give you a basic idea of the difference between them!

### Examples (if any): 

