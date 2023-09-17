
#incubator 
###### upstream: 

### Strong Consistency with Synchronous Replication:

***Strong consistency** ensures that a system provides the illusion that there is only a single, up-to-date copy of the data at all times. A **synchronous system** provides the illusion of immediate and synchronized updates across all replicas or nodes involved in the system*

- In a **strongly consistent** system with **synchronous replication**, if a write operation is not replicated to all other nodes, the system may reject the operation and return an error to the user. This ensures that the system maintains strict consistency, and any read operation following the write will always see the updated data across all nodes.
-  In this case, the user will receive an error message indicating that the write operation could not be completed successfully due to a failure to replicate the data to all nodes. This guarantees that the user will not observe inconsistent or out-of-sync data across different parts of the system.

#### Examples 

1. **Google Cloud Spanner**: Google Cloud Spanner is a globally distributed relational database service offered by Google Cloud Platform. It provides strong consistency guarantees across regions through the use of synchronous replication and the TrueTime technology, which provides a synchronized global clock. Spanner allows applications to perform strongly consistent reads and writes across its globally distributed architecture.

3. **CockroachDB**: CockroachDB is an open-source distributed SQL database designed to provide strong consistency and high availability. It uses a distributed consensus protocol called Raft and employs synchronous replication to ensure that data is consistently replicated across all nodes in the cluster. CockroachDB offers strong consistency guarantees and ACID transactions across a distributed environment.

5. **RDS**: AWS offers services like Amazon RDS (Relational Database Service) and Amazon Aurora that provide strong consistency within a single region by default. They employ synchronous replication techniques to ensure data durability and availability *but may not guarantee strong consistency across regions*.To achieve strong consistency with synchronous replication in AWS, you may need to combine multiple services or utilize custom architectures based on your specific requirements. (some architecture examples include **Multi-Availability Zone (Multi-AZ) Deployment**, **Cross-Region Replication**, and **Custom Distributed Systems**) It's recommended to consult the AWS documentation and explore different service offerings and their consistency guarantees to determine the best approach for achieving strong consistency in your particular use case.



### Eventual Consistency with Asynchronous Replication:

*A system with **eventual consistency** and **asynchronous replication** provides the illusion of relaxed consistency and eventual convergence of data across replicas. In this type of system, write operations do not wait for replication to complete before returning a response to the client, and data updates are propagated asynchronously to replicas. As a result, there may be temporary inconsistencies or divergence in data observed across replicas.*

-  In an **eventually consistent** system with **asynchronous replication**, the write operation can be acknowledged and considered successful even if it hasn't been replicated to all other nodes at that exact moment. The replication process occurs asynchronously in the background.
-  In this scenario, the user may not receive an error immediately, as the system prioritizes availability and responsiveness. However, if the user performs a subsequent read operation, there is a possibility of observing stale or outdated data until the replication is complete. The user may experience eventual consistency, where the data across nodes eventually becomes consistent over time

#### Examples 

1. **Amazon S3 (Simple Storage Service)**: Amazon S3 is a highly scalable object storage service provided by AWS. It uses eventual consistency to replicate data across multiple availability zones. When new data is written or updated, it may take some time for all replicas to reflect the changes. S3 provides strong consistency for overwrite PUTS and DELETES within a single bucket, but eventual consistency for new object creations and updates across different buckets.

2. **Azure Blob Storage**: Azure Blob Storage is a scalable and durable object storage service offered by Microsoft Azure. It employs eventual consistency for replicating data across multiple regions and availability zones. Similar to Amazon S3, Blob Storage ensures strong consistency within a single container but provides eventual consistency when accessing data across containers.

3. **Google Cloud Storage**: Google Cloud Storage is a scalable and highly available object storage service provided by Google Cloud Platform. It uses eventual consistency for data replication across multiple regions. Updates made to objects may take some time to propagate across all replicas, resulting in eventual consistency guarantees.

4. **Apache Kafka**: Apache Kafka is a distributed streaming platform that provides event streaming capabilities with high throughput and fault tolerance. Kafka follows an eventual consistency model, where data is replicated asynchronously across multiple brokers within a Kafka cluster. This allows for high availability and scalability but may result in data inconsistencies for a brief period until replication is complete.

5. **MongoDB Atlas**: MongoDB Atlas is a fully managed cloud database service provided by MongoDB. It uses asynchronous replication to replicate data across multiple nodes and regions. MongoDB offers tunable consistency levels, including eventual consistency, to balance performance and consistency requirements based on the application's needs.


### Weak Consistency: 

- Weak consistency provides the weakest level of consistency guarantees. 
- In this model, there are no strict ordering or synchronization guarantees across different replicas. 
- Updates to data may propagate asynchronously, and different replicas may have inconsistent views of the data for a period of time. 
- Weak consistency allows for high availability and scalability but may result in data conflicts or inconsistencies during concurrent updates.

#### Examples 

1. **Amazon DynamoDB**: Amazon DynamoDB is a fully managed NoSQL database service provided by AWS. It employs a form of eventual consistency known as "last writer wins." In DynamoDB, updates to a specific data item may propagate asynchronously across replicas, and conflicting updates may result in eventual convergence. DynamoDB allows developers to specify the desired consistency level for each read operation, enabling a trade-off between consistency and availability.

2. **Cassandra on DataStax Astra**: DataStax Astra is a cloud-native database-as-a-service (DBaaS) platform based on Apache Cassandra. Cassandra is designed with a focus on high availability and partition tolerance, and it offers tunable consistency levels. By adjusting the consistency levels, developers can balance the trade-off between consistency and availability based on their application requirements.

3. **Riak**: Riak is a distributed NoSQL database that provides high availability and fault tolerance. It employs a "last writer wins" conflict resolution strategy and offers tunable consistency controls. Riak allows developers to choose between strong consistency, eventual consistency, or other trade-offs along the consistency spectrum.

4. **Couchbase**: Couchbase is a distributed NoSQL database designed for high availability and scalability. It provides tunable consistency levels, allowing developers to configure the desired level of consistency based on their application needs. Couchbase offers different consistency models, including eventual consistency, session consistency, and strong consistency.

5. **Redis**: Redis is an in-memory data structure store that supports various data types and is commonly used as a cache or database. While Redis is known for its low latency and high throughput, it typically offers weak consistency guarantees. Replication in Redis is asynchronous, and updates made to the master node may take some time to propagate to the replica nodes, resulting in eventual consistency.



### Read-your-Writes Consistency:

*A system with **read-your-writes** consistency provides the illusion that a client will always observe its own writes immediately after performing them. In other words, any read operation performed by a client will reflect the most recent write operation performed by that same client.*

- Read-your-writes consistency guarantees that a client will always observe its own writes immediately after performing them. 
- Any subsequent read operation by the same client will see the most recent version of the data it wrote
- This consistency model does not guarantee that other clients will observe the latest data immediately.

#### Examples: 

1. **Google Cloud Firestore**: Google Cloud Firestore is a serverless, NoSQL document database provided by Google Cloud Platform. It offers read-your-writes consistency, ensuring that any read operation performed by a client after a write operation will see the updated data immediately. Firestore provides strong consistency guarantees within a single document or collection when using transactions or document writes.

2. **Microsoft Azure Cosmos DB**: Azure Cosmos DB is a globally distributed, multi-model database service provided by Microsoft Azure. Cosmos DB supports multiple consistency levels, including strong consistency, bounded staleness, session consistency, and eventual consistency. By selecting the "Session" consistency level, clients can achieve read-your-writes consistency within a session, ensuring that their own writes are immediately visible in subsequent read operations.

3. **MongoDB Atlas**: MongoDB Atlas is a fully managed cloud database service provided by MongoDB. MongoDB offers read-your-writes consistency as part of its default behavior. When using a single replica set deployment, MongoDB ensures that subsequent reads by a client will always observe their own writes, providing read-your-writes consistency at the document level.

4. **FaunaDB**: FaunaDB is a distributed, serverless, and globally replicated transactional database. It offers strong consistency by default, including read-your-writes consistency. FaunaDB guarantees that any read operation performed by a client will always see its own writes immediately, maintaining strong consistency and ordering guarantees.



### Monotonic Consistency:

*A system with **monotonic** consistency provides the illusion that a client will always observe data changes in a consistent and non-decreasing order. In other words, if a client reads the same data item multiple times, it will not see an older version of the data aka [[Stale Data]]*

- Ensures that if a client reads the same data item multiple times, it will not observe a "stale" or older version of the data 
- Once a client has observed a particular order of updates to a data item, it will not see those updates in a different order. 
- This consistency model preserves causality and prevents retrograde or out-of-order updates.

#### Examples: 

1. **Amazon DynamoDB**: Amazon DynamoDB, a fully managed NoSQL database service provided by AWS, offers a form of monotonic consistency. DynamoDB guarantees that once a client reads a particular version of an item, it will not see a previous version in subsequent reads. This ensures monotonic consistency within a single item or document.

2. **Google Cloud Spanner**: Google Cloud Spanner is a globally distributed and strongly consistent relational database service provided by Google Cloud Platform. Spanner provides monotonic consistency guarantees by preserving causality across transactions. Once a client observes a particular set of transactions or operations, it will not observe those same operations in a different order or see older versions of the data in subsequent reads.

3. **FaunaDB**: FaunaDB is a distributed, serverless, and globally replicated transactional database. It offers strong consistency guarantees by default, which includes monotonic consistency. FaunaDB ensures that clients will not observe a stale or older version of the data when reading the same item multiple times.


### Session Consistency:

- Provides consistency guarantees within the context of a client session
- All reads and writes performed by a client within a session are guaranteed to be seen by subsequent operations within the same session
- Different sessions may observe inconsistent views of the data.

#### Examples 

1. **Azure Cosmos DB**: Azure Cosmos DB is a globally distributed, multi-model database service provided by Microsoft Azure. Cosmos DB offers multiple consistency levels, including session consistency. When using session consistency, all read and write operations performed within a session are guaranteed to be seen in a consistent order by subsequent operations within the same session. This ensures session-level consistency while allowing for weaker consistency guarantees across different sessions.

2. **Apache Cassandra on DataStax Astra**: DataStax Astra is a cloud-native database-as-a-service (DBaaS) platform based on Apache Cassandra. Cassandra provides tunable consistency levels, including session consistency. With session consistency, all read and write operations performed within a session are guaranteed to see the most recent version of the data. However, different sessions may observe different versions of the data due to the eventual nature of Cassandra's replication process.

3. **MongoDB Atlas**: MongoDB Atlas is a fully managed cloud database service provided by MongoDB. MongoDB offers session-level consistency through its transactions feature. By encapsulating multiple operations within a transaction, MongoDB ensures that all read and write operations performed within a transaction will observe a consistent snapshot of the data. This provides session consistency guarantees within the context of a transaction.
