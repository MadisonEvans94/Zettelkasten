#incubator 
###### upstream: [[wiki]], [[Algorithms]]

### Origin of Thought:
- Exploring how wiki works will strengthen overall understanding of graphs and their use cases 

### Underlying Question: 
- What types of graph algorithms are used in wikis and why? 

### Solution/Reasoning: 

- **[[shortest path]] algorithms**: 
	- used to find the shortest path between nodes 
	- could be used to find the shortest sequence of hyperlinks between 2 articles 
	- Djikstras, Bellman-Ford 
- **[[PageRank]] algorithms**: 
	- used to rank websites in their search engine results, and a big consideration in [[SEO]]
	- can be used to rank articles based on their importance, inferred from how many other articles link to them
	- the more links and the more important those linked pages are, the higher the rank 
- **[[Community Detection]] algorithms**:
	- used to identify the formation of clusters by use of [[centrality measures]]
	- in the Obsidian context, think of MOC detectors 
- **[[Link Prediction Algorithms]]**: 
	- as the name implies, these are algorithms used to predict the formation of future links based on node proximity 

### Examples (if any): 


### Additional Questions (if any): 
- [[Graph Algorithms for Social Networks]]

