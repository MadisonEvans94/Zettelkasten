#incubator 

###### upstream: 

### Description

The **CAP Theorem**, also known as **Brewer's Theorem**, states that it is impossible for a distributed data store to simultaneously provide more than two out of the following three guarantees:

- **Consistency:** Every read receives the most recent write or an error. In other words, having a single up-to-date copy of the data.
- **Availability:** Every request receives a (non-error) response, without the guarantee that it contains the most recent write. 
- **Partition Tolerance:** The system continues to operate despite an arbitrary number of messages being dropped (or delayed) by the network between nodes.

In simple terms, the CAP theorem states that in the presence of a network partition, one has to choose between consistency and availability.

**Understanding CAP Theorem Components:**

1. **Consistency (C):** Consistency ensures that all clients see the same data at the same time, no matter which node they connect to. For this to happen, whenever data is written to one node, it must be instantly forwarded or replicated to all the other nodes in the system before the write is deemed ‘successful.’ see [[Consistency in Depth]] for more details
		*basically, whatever changes you make won't be implemented until the change has been replicated to all other nodes*

3. **Availability (A):** Availability ensures that the system remains operational 100% of the time. This means that every request received by a non-failing node in the system must result in a response. Even if one or more nodes are down, every request must still be processed. This is achieved by replicating the data across different servers. 

4. **Partition Tolerance (P):** Partition Tolerance is the ability to maintain service despite the network failing to correctly transmit messages between nodes (this could be due to network outages, or delays). If a network between nodes breaks, the system should still function. 

### Tradeoffs in CAP Theorem: 

The CAP theorem states that you can't achieve all three of these guarantees at the same time, you have to pick two:

- **CA - Consistency and Availability:** This is usually seen in single-node databases. There is no partition tolerance because the system is on a single node, meaning there are no partitions. This system offers consistency by ensuring that every transaction brings it from one valid state to another, and availability by guaranteeing that every request gets a response about whether it was successful or failed.

- **CP - Consistency and Partition Tolerance:** This is usually seen in multi-node databases where achieving consistency is a must and partition tolerance is inevitable. However, availability will be affected. If a partition occurs between nodes, the system will block any incoming requests, leading to downtime until the partition is resolved.

- **AP - Availability and Partition Tolerance:** This is usually seen in systems where the ability to continue reading and writing data is more important than consistency. This is often seen in real-time data processing systems where if there’s a partition between nodes, all nodes remain available but those at the wrong end of a partition might return an older version of data than others.
- 
**Example systems:**
- Apache Cassandra, Riak, CouchDB are examples of AP systems.
- HBase, MongoDB, and Redis are examples of CP systems.
- Traditional RDBMS like MySQL, Oracle are examples of CA systems (but without partition tolerance, they can't truly work in a distributed setting).

See [[Cap Theorem Cloud Examples]] for more Examples

Please note that these are somewhat fluid: many of these systems offer some configurability that lets you adjust where on the CAP spectrum you want to be. Additionally, real-world systems often make additional compromises, effectively providing a subset of each CAP property.

