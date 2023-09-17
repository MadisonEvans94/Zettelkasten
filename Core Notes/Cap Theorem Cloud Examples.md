#incubator 
###### upstream: 

### Architecture examples: 

*What is an example of a model that provides **Consistency** and **Availability**?* 

Many cloud services that prioritize consistency and availability to varying degrees based on their design and use cases. Here are some examples:

1.  **Amazon DynamoDB**: Amazon DynamoDB is a fully managed NoSQL database service provided by AWS. It offers high availability and durability with automatic multi-region replication. DynamoDB allows you to configure the consistency requirements for reads, providing options for strong consistency or eventual consistency.
    
2.  **Google Cloud Spanner**: Google Cloud Spanner is a globally distributed and strongly consistent relational database service provided by Google Cloud Platform. It offers both consistency and availability across multiple regions through the use of synchronous replication and TrueTime, a global clock synchronization technology.
    
3. **Microsoft Azure Cosmos DB**: Azure Cosmos DB is a globally distributed, multi-model database service provided by Microsoft Azure. It provides tunable consistency levels, allowing you to choose between strong consistency, bounded staleness, session consistency, or eventual consistency based on your application requirements.
    
4. **Apache Cassandra on DataStax Astra**: DataStax Astra is a cloud-native database-as-a-service (DBaaS) platform based on Apache Cassandra. It offers high availability and fault tolerance with automatic multi-region replication. Cassandra provides tunable consistency levels, allowing you to configure the desired trade-off between consistency and availability.
    
5. **CockroachDB**: CockroachDB is a distributed SQL database that offers consistency and availability guarantees. It provides strong consistency by default, allowing linearizable reads and writes, while also ensuring high availability through automatic replication and automatic data rebalancing.
    

It's important to note that while these services prioritize consistency and availability, they may still exhibit trade-offs or limitations in certain scenarios, such as during network partitions or in the face of extreme network latency or failures. The specific choice of cloud service would depend on your application requirements and the desired balance between consistency and availability.

*What is an example of a model that provides **Consistency** and **Partition Tolerance**?* 

In the context of the CAP theorem, cloud services that prioritize consistency and partition tolerance often sacrifice availability in certain situations. Here are some examples of cloud services that prioritize consistency and partition tolerance:

1. **Google Cloud Bigtable**: Google Cloud Bigtable is a distributed, high-performance NoSQL database service provided by Google Cloud Platform. It prioritizes consistency and partition tolerance by using a distributed storage system based on the Google File System (GFS) and the Chubby lock service. However, it may experience reduced availability during network partitions.
    
2. **Apache HBase**: Apache HBase is an open-source, distributed, column-oriented database built on top of Hadoop and HDFS. It provides strong consistency guarantees and partition tolerance. However, availability may be affected during network partitions or when nodes become unreachable.
    
3. **Apache ZooKeeper**: Apache ZooKeeper is a centralized coordination service that provides distributed synchronization and configuration management. It is designed to prioritize consistency and partition tolerance. ZooKeeper ensures that updates are ordered and maintains a consistent view of the distributed system, but it may have reduced availability during network partitions.
    
4. **etcd**: etcd is an open-source distributed key-value store that provides reliable storage for distributed systems. It is designed to prioritize consistency and partition tolerance using the Raft consensus algorithm. However, during network partitions or when a majority of nodes become unavailable, etcd may experience reduced availability.
    
5. **Consul**: Consul is a distributed service mesh and key-value store designed for service discovery, configuration, and coordination. It prioritizes consistency and partition tolerance by using the Raft consensus algorithm. However, availability can be affected during network partitions or when a majority of nodes become unavailable.
    

These services prioritize consistency and partition tolerance, which means they strive to provide strong consistency guarantees even in the face of network partitions. However, they may experience reduced availability during such scenarios. It's important to consider the trade-offs and choose the appropriate service based on your application requirements and tolerance for potential availability impacts.


*What is an example of a model that provides **Partition Tolerance** and **Availability**?* 

In the context of the CAP theorem, cloud services that prioritize partition tolerance and availability often sacrifice strict consistency. Here are some examples of cloud services that provide partition tolerance and availability:

1. **Amazon S3 (Simple Storage Service)**: Amazon S3 is a scalable object storage service provided by AWS. It offers high availability and durability for storing and retrieving any amount of data. While S3 provides strong consistency for overwrite PUTS and DELETES within a single bucket, it may exhibit eventual consistency for new object creations and updates in certain situations.

2. **Microsoft Azure Blob Storage**: Azure Blob Storage is a scalable object storage service provided by Microsoft Azure. It offers high availability, durability, and scalability for storing large amounts of unstructured data. Similar to Amazon S3, Azure Blob Storage provides strong consistency within a single container but may have eventual consistency in some cases.

3. **Google Cloud Storage**: Google Cloud Storage is a scalable and highly available object storage service provided by Google Cloud Platform. It offers high durability and availability for storing and retrieving data. Google Cloud Storage provides strong consistency for read-after-write operations and strong list consistency for bucket metadata.

4. **Microsoft Azure Cosmos DB**: Azure Cosmos DB is a globally distributed, multi-model database service provided by Microsoft Azure. It is designed to be highly available and partition tolerant. While offering multiple consistency options, Cosmos DB allows you to prioritize availability and partition tolerance by choosing weaker consistency models like eventual consistency or bounded staleness.

5. **Apache Kafka**: Apache Kafka is a distributed streaming platform that provides fault-tolerant, high-throughput, and low-latency event streaming. Kafka is designed to be highly available and partition tolerant by leveraging replication and distributed commit logs. While it guarantees partition tolerance and availability, it provides at-least-once message delivery semantics, which means that duplicate messages may occur.

These cloud services prioritize partition tolerance and availability, allowing them to function reliably even in the presence of network partitions or failures. However, it's important to note that they may exhibit eventual consistency or weaker consistency models to achieve high availability. The choice of service would depend on your specific requirements, such as the type of data you need to store or the nature of your application's data access patterns.