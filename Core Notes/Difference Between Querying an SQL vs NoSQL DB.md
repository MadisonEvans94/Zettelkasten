#incubator 
###### upstream: [[Databases]]

### Origin of Thought:
- Strenghten understanding of databse querying by understanding the difference across different db types 

### Underlying Question: 
- What exactly does it mean to query a database and how is querying different between SQL and NoSQL db?

### Solution/Reasoning: 
**Database Querying**

- Database querying is a process of requesting specific data from a database
- Involves writing commands or queries in a language that the database system can understand
- Performs operations such as insert, update, delete and retrieve data.

**SQL Databases**

- SQL (Structured Query Language) databases are relational databases that use structured query language (SQL) for defining and manipulating the data
- In SQL databases, data is stored in tables and these tables are related to each other. Examples of SQL databases include MySQL, PostgreSQL, Oracle Database, and SQL Server.
- The primary characteristic of SQL databases is that they follow [[ACID]] properties (Atomicity, Consistency, Isolation, Durability)
- SQL databases use a schema, which is a structured blueprint of how data is organized.
- SQL databases use SQL for querying. An example of a SQL query to retrieve data might look like this:


```sql
SELECT * FROM Customers WHERE Country='Germany';
```


**NoSQL Databases**

- NoSQL databases were created in response to the limitations of SQL databases, particularly regarding scale, replication, and unstructured storage
- NoSQL databases can store and process data in real-time and are highly scalable
- Examples include MongoDB, Apache Cassandra, and Google's Bigtable.
- NoSQL databases do not have a standard querying language and the query depends on the type of NoSQL database (document, key-value, columnar, graph)
- They typically offer APIs and other interfaces for interacting with the data, and the data structure can vary greatly.

For a document-oriented NoSQL database like MongoDB, a query might look like:


```js
db.customers.find({ "country": "Germany" })
```


This query would perform a similar operation to the SQL query above, returning all documents in the 'customers' collection where the 'country' field is 'Germany'.

This query would return all data from the `Customers` table where the `Country` is `Germany`.

**Difference Between SQL and NoSQL Querying**

In terms of querying, SQL and NoSQL databases differ significantly:

1.  **Query Language:** SQL uses the [[Structured Query Language]] (SQL), which is highly structured and powerful with a standardized syntax. NoSQL querying varies by the database and its structure; it does not have a standard querying language.
    
2.  **Schema:** SQL databases require data to be stored in tables with a predefined schema, so queries must adhere to this structure. NoSQL databases are often schema-less, allowing for flexible data structures, which can affect the way data is queried.
    
3.  **Complexity:** SQL databases are great for complex queries, as SQL is a very powerful language. NoSQL databases can struggle with complex queries, and often the logic needs to be handled in the application code.
    
4.  **Relations:** SQL databases are relationally oriented, and this is reflected in the query language, which allows for sophisticated joining of data across tables. NoSQL databases do not typically have the concept of joins in their querying.
    
5.  **Scalability:** NoSQL databases are designed for scalability, and querying is often optimized for large, distributed environments. SQL databases and their query language are more centralized, with optimizations primarily for single-server environments, although modern SQL databases have made significant advances in distributed operations.
    

In conclusion, the choice between SQL and NoSQL - and therefore the way you will perform queries - largely depends on the specific requirements of your project, such as the complexity of the data and queries, the need for scalability, and the type of data structure you'll be dealing with.

### Examples (if any): 

### Additional Questions: 

- [[Why NoSQL databases can store and process data in real-time with high scalability]]

