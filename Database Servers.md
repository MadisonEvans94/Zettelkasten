#seed 
upstream:

---

**video links**: 

---


A **database server** is a server which houses a database application that provides database services to other computer programs or to computers, as defined by the client–server model.

Here are some key points to understand about database servers:

## Functions of a Database Server

1. **Data Management**: The primary function of a database server is managing and providing access to a *centralized* database. The server takes care of storing, retrieving, and updating data in a database [[CRUD]].

2. **Concurrency Control**: A database server can handle multiple requests from clients at the same time. It manages *concurrent* access to the data to maintain data integrity and consistency.

3. **Security**: A database server provides security measures to protect data. It controls who can access the data and what operations they can perform.

4. **Backup and Recovery**: The server ensures the reliability of data by providing backup and recovery features. This helps protect data from accidental loss.

## Database Servers vs Other Servers

Database servers are specialized servers that are optimized for tasks related to databases. Other types of servers, like **web servers** or **file servers**, have different purposes. Web servers, for instance, are optimized for serving web pages and web applications, while file servers are designed to store, share and manage access to files in a network environment. *see [[Server Optimization Guide]] for more*

## Multiple Databases

Yes, a single database server can manage multiple databases. This is similar to how you can have multiple connections in MySQL Workbench. Each database is a separate set of data, and they can be used by different applications. For example, a company may have one database for their customer information and another for their product inventory.

## Knowledge as an Engineer

As an engineer, here are some things you should know about database servers:

1. **[[Understand Different Types of Databases]]**: Databases come in many types, such as relational databases (like MySQL, PostgreSQL), NoSQL databases (like MongoDB, Cassandra), and in-memory databases (like Redis). Understand their differences and use cases.

2. **[[Database Design]]**: You should understand how to design a database schema. This includes knowing how to normalize data, define relationships, and set up indexes.

3. **[[SQL]]**: **Structured Query Language (SQL)** is the standard language for interacting with relational databases. You should know how to write SQL queries to create, read, update, and delete data.

4. **[[Performance Tuning]]**: Databases can become bottlenecks if they are not well-optimized. You should understand how to tune a database for performance. This can involve optimizing queries, setting up indexes, and configuring the database server settings.

5. **Backup and Recovery**: Understand how to backup data and restore data from backups. This is crucial for preventing data loss.

6. **Security**: Know how to secure a database. This includes setting up user permissions, encrypting data, and protecting against SQL injection attacks.

7. **Scaling**: As your application grows, you might need to scale your database. Understand the difference between vertical and horizontal scaling, and the trade-offs between them.

8. **Replication and Sharding**: Replication is about copying data from one database server to others to increase data availability. Sharding is about splitting a larger database into smaller ones to improve performance. These are advanced topics but are very useful to know when dealing with large-scale applications.
   
Understanding these points will provide a solid foundation in working with database servers. However, the depth of your knowledge will grow as you gain more hands-on experience working with different types of databases and addressing various application needs.

