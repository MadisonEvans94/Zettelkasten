#seed 
upstream: [[Data Structures]]

### Introduction 

A **database** is a systematic collection of data. They support the storage and manipulation of data. Databases make data management easy.

### What is a Database? <a name="what-is-database"></a>

A database is an organized collection of data stored and accessed electronically. It is designed to hold data, organize it in a meaningful way, and allow manipulation of the data in various ways.

### Types of Databases <a name="types-of-databases"></a>

*There are mainly four types of databases:*

#### 1. **[[Relational Database]]**: 

Also known as a SQL database, it organizes data into tables. Example: **MySQL**, **Oracle**, **PostgreSQL**.

#### 2. **[[Object Oriented Databases]]**: 

see [[object oriented databases vs key value stores]]

#### 4. **[[Hierarchical database]]**: 
In this type of database, the data is organized into a tree-like structure.

#### 5. **[[Network database]]**: 
This is a type of database model wherein each record not only has multiple parents but also multiple children. Example: Integrated Data Store (IDS)

### Database Management Systems (DBMS) <a name="dbms"></a>

A **Database Management System (DBMS)** is the software that interacts with end users, applications, and the database itself to capture and analyze data. 

A general-purpose DBMS allows the definition, creation, querying, update, and administration of databases. Examples include **MySQL**, **PostgreSQL**, **MongoDB**, **Oracle Database**, and **SQL Server**.

### Components of a Database <a name="components-of-database"></a>

*A database system typically includes the following components:*

**Tables**: 
This is where the data is stored.

**Queries**: 
A way to retrieve information from a database.

**Forms**: 
A way to enter data into a table or to get data from tables.

**Reports**: 
It pulls data from tables to show it in a structured way.

**Macros**: 
These are automated scripts.

### Database Design 

**Database design** is the process of producing a detailed data model of a database. This data model contains all the needed logical and physical design choices and physical storage parameters needed to generate a design.

A good design is important for efficiency, reliability, and accuracy. This involves understanding the business needs, creating a logical model, transforming to a physical model, and then implementing the model. 

>See [[ACID]] principles for more

>See [[Database Application Development Methodology]] for more
### Normalization <a name="normalization"></a>

**Normalization** is a method to eliminate the data redundancy and dependency by organizing the fields and tables of a database. It involves decomposing a table into less redundant tables without losing information.

### SQL <a name="sql"></a>

**[[SQL (Structured Query Language)]]** is a standard language for storing, manipulating, and retrieving data in databases. SQL is used to communicate with a database and SQL statements are used to perform tasks such as update data on a database, or retrieve data from a database.

### NoSQL <a name="nosql"></a>

[[NoSQL]] databases (non-relational) provide a mechanism for storage and retrieval of data that is modeled in means other than the tabular relations used in relational databases. These databases are increasingly used in big data and real-time web applications.

### Database Security <a name="database-security"></a>

**Database security** refers to the collective measures used to protect and secure a database or database management software from illegitimate use and malicious threats and attacks.

### Cloud Databases <a name="cloud-databases"></a>

**Cloud databases** are databases that run on cloud computing platforms that deliver computing as a service instead of a product. They can run on traditional database software such as MySQL, or on software technology specifically designed for cloud computing such as Amazon's DynamoDB.

### Additional Resources <a name="additional-resources"></a>

- [Database Design Tutorial](https://www.tutorialspoint.com/dbms/dbms_tutorial.htm)
- [SQL Tutorial](https://www.w3schools.com/sql/)
- [NoSQL Databases](https://www.tutorialspoint.com/nosql/index.htm)
