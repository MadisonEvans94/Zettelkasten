
#incubator 
upstream: [[Cloud Computing and Distributed Systems]], [[RDS]]

---

**video links**
- [connect to RDS](https://www.youtube.com/watch?v=6Nt-Jl3CzxE&ab_channel=RhysDent)

---
## Introduction 

When working in a **distributed system** where your application server (i/e running on AWS Elastic Beanstalk) and your database (i/e PostgreSQL on RDS) are **separate**, `Sequelize` still plays a similar role.

## Under the hood
`Sequelize` translates your high-level JavaScript code into **SQL queries** that your database can understand. 

### Process Flow: 

It's important to note that the request sent to and from database is not an HTTP request, but a database query sent over a **database connection**. See [[The difference between database connection and http request]] for more detail 

#### 1. JS to SQL Translation
`Sequelize` translates the `create` method into an SQL query (e.g., `INSERT INTO "Users" ("email", "password") VALUES ('user@example.com', 'password') RETURNING *;`)

#### 2. CRUD action sent to database
This SQL query is sent to your database (PostgreSQL on RDS) over a **database connection**.

#### 3. Database Server Execution
The PostgreSQL database executes the SQL query and returns the result.

#### 4. SQL to JS Translation
`Sequelize` **translates** the result back into JavaScript and returns it to your application.

## Configuration

In order to set up `Sequelize` to connect to your PostgreSQL database on RDS from your application on Elastic Beanstalk, you need to provide the **connection details** to `Sequelize`. This typically includes the **hostname** (or IP address) of the RDS instance, the **port number**, the **database name**, and the **username** and **password** for the database.

*Here's an example:*

```javascript
const { Sequelize } = require('sequelize');

const sequelize = new Sequelize('database', 'username', 'password', {
  host: 'my-rds-instance.amazonaws.com',
  port: 5432,
  dialect: 'postgres',
  dialectOptions: {
    ssl: {
      require: true,
      rejectUnauthorized: false
    }
  }
});
```

*In the above example:*

- Replace `'database'`, `'username'`, and `'password'` with the name of your database and the credentials for your database user.
- Replace `'my-rds-instance.amazonaws.com'` with the hostname of your RDS instance. You can find this in the RDS console in AWS.
- `5432` is the default port for PostgreSQL. If you're using a different port, replace `5432` with your port number.
- The `ssl` options are required for connecting to RDS PostgreSQL instances. This ensures that the connection to the database is encrypted.

It's also best practice to store sensitive data like your database credentials in environment variables, rather than hardcoding them into your application. AWS Elastic Beanstalk provides a way to set environment variables for your application. See [[Understanding Environment Variables in AWS]] and [[Understanding Environment Variables in Node.js]]for more

With `Sequelize `set up this way, your application on Elastic Beanstalk can communicate with your PostgreSQL database on RDS.

## Final Thoughts

In a distributed system like this, it's crucial to consider the potential for network latency and connection issues between your application and your database. It's also important to manage database connections effectively, as there are limits to how many simultaneous connections your database can handle.

Finally, be aware that running database operations can be time-consuming. To prevent these operations from blocking other tasks in your Node.js application, always use `async` and `await` (or Promises) when calling Sequelize methods, to ensure these operations are performed asynchronously.



