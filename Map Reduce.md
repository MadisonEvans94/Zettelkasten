#seed 
upstream: [[Big Data]]

---

**video links**: 

---

# Brain Dump: 


--- 

**MapReduce** is a programming paradigm that allows for the distributed processing of large data sets across clusters of computers. It was popularized by Google in a research paper in 2004, and it's the foundation for many large-scale data processing tasks, such as those found in big data applications. The paradigm itself is influenced by functional programming concepts and consists of two main phases: the Map phase and the Reduce phase.

## Core Concepts

1. **Map Phase**: In the Map phase, the input data is divided into smaller sub-parts, called "splits." A "mapper" function is applied to each split. The function processes the data and produces a set of intermediate key-value pairs.

$$
\text{Map:} (k_1, v_1) \rightarrow \text{list}(k_2, v_2)
$$

2. **Shuffle Phase**: After the Map phase, there's often a "Shuffle" phase, which sorts and groups the key-value pairs based on the keys. All values for the same key are grouped together.

3. **Reduce Phase**: In the Reduce phase, a "reducer" function processes each group of values that share the same key and combines them into a smaller set of values, usually by performing some form of aggregation.

$$
\text{Reduce:} (k_2, \text{list}(v_2)) \rightarrow \text{list}(k_3, v_3)
$$

## Execution Flow

1. **Input Splits**: The input data is divided into smaller chunks called "splits."
2. **Map Function**: Each split is processed by a Map task, generating a set of intermediate key-value pairs.
3. **Shuffling**: The key-value pairs are sorted and grouped by key.
4. **Reduce Function**: Each group of values with the same key undergoes a Reduce task, where they are further processed and aggregated.

## Fault Tolerance

MapReduce is designed to be fault-tolerant. If a node fails during the Map or Reduce phases, the job scheduler can reroute the task to another node. This is crucial when dealing with large-scale data processing where hardware failures are not uncommon.

