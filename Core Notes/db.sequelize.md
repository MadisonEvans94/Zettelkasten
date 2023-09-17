#incubator 
upstream: 

---

**video links**: 

---
## `db.sequelize` breakdown 

The `db.sequelize` object represents the instance of `Sequelize` that is connected to the **database**. It provides several methods and properties that can be used to interact with the database and perform various operations. 

*In other words, `db` is your object that represents all your models, and `db.sequelize` holds all your **CRUD** capabilities*

### Important Methods
*Here* are some important methods of `db.sequelize`:

#### `authenticate()`
Tests the connection to the database by authenticating. It returns a Promise that resolves if the authentication is successful, and rejects with an error if it fails.
```js
db.sequelize.authenticate()
  .then(() => {
    console.log('Connection has been established successfully.');
  })
  .catch((error) => {
    console.error('Unable to connect to the database:', error);
  });
```

#### `sync(options)`
Synchronizes all defined models with the database by creating the corresponding tables if they do not exist. This method returns a Promise that resolves when the synchronization is complete.
```js
db.sequelize.sync()
  .then(() => {
    console.log('Models synchronized with the database.');
  })
  .catch((error) => {
    console.error('An error occurred while synchronizing models:', error);
  });
```

#### `query(sql, [options])`
Executes a raw SQL query on the database. It allows you to perform custom queries that are not covered by the `Sequelize `query methods. It returns a Promise that resolves with the query result.
```js 
db.sequelize.query('SELECT * FROM Users', { type: db.sequelize.QueryTypes.SELECT })
  .then((users) => {
    console.log(users);
  })
  .catch((error) => {
    console.error('An error occurred while executing the query:', error);
  });
```

#### `transaction([options], [autoCallback])`
Starts a transaction for performing multiple database operations as a single unit of work. It allows you to ensure atomicity and consistency in a series of operations. This method returns a Promise that resolves with a transaction object.
```js
db.sequelize.transaction((transaction) => {
  return db.User.create({ name: 'John Doe', email: 'johndoe@example.com' }, { transaction })
    .then(() => {
      return db.Course.create({ name: 'Node.js Basics' }, { transaction });
    })
    .then(() => {
      // Other operations within the transaction
      // ...
    })
    .catch((error) => {
      console.error('An error occurred within the transaction:', error);
      throw error; // Rollback the transaction
    });
})
  .then(() => {
    console.log('Transaction committed successfully.');
  })
  .catch((error) => {
    console.error('Transaction failed:', error);
  });
```

#### `close()`
Closes all connections used by the `Sequelize` instance.
```js
db.sequelize.close()
  .then(() => {
    console.log('Connection closed successfully.');
  })
  .catch((error) => {
    console.error('An error occurred while closing the connection:', error);
  });
```

These are just a few examples of the methods available on the `db.sequelize` object. `Sequelize` provides a rich API for interacting with the database, including methods for **querying**, **creating**, **updating**, and **deleting** records.

---

## TLDR

**`db.sequelize`** is the object that holds all your **[[CRUD]]** capabilities. It holds all the *action* verbs that can be used on the database instance 