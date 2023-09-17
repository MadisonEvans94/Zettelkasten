#evergreen1 
upstream: [[Object Relational Mapping (ORM)]]

**`Sequelize`** is a promise-based `Node.js` [[Object Relational Mapping (ORM)]] for **Postgres**, **MySQL**, **SQLite** and **Microsoft SQL Server**. It features solid transaction support, relations, eager and lazy loading, and read replication.

---
## The `Sequelize` Constructor and the `sequelize` Object

At the heart of this library is the **`Sequelize`** class, which we often instantiate with the `new` keyword to create a `sequelize` object. This object forms the base of our interactions with our database. *Sequelize class is a factory and the object it creates gives us access to the database declared in the argument of the constructor*

To understand the `Sequelize` constructor and the resulting object in more depth, let's break it down... 

### The `Sequelize` Constructor

The `Sequelize` constructor is a class that takes several parameters to establish a connection to a database. The parameters include **`database`**, **`username`**, **`password`**, and an **`options`** object that has various optional settings. 

*Here's an example:*

```javascript
const sequelize = new Sequelize('database', 'username', 'password', {
  host: 'localhost',
  dialect: 'mysql',
});
```

#### Parameters

- `'database'`: The name of your database.
- `'username'`: The username for your database.
- `'password'`: The password for your database.
- `options`: An object with additional configuration settings. 

for details on connecting to the cloud, see [[Working with Sequelize in a Distributed Cloud System]]

##### The `options` object 

*...can have several properties:*

- `host`: The host of your database. Default is `'localhost'`.
- `port`: The port of the host of the database 
- `dialect`: The dialect of your database. `Sequelize` supports `mysql`, `sqlite`, `postgres`, `mssql`.
- `dialectOptions`: options for the dialect such as `ssl`
- `pool`: Optional configuration for pooling connections.
- `storage`: Only for SQLite, the storage engine to use.
- `logging`: Function that gets executed every time `Sequelize` would log something.
- `timezone`: The timezone used when following date data types.
- 
*the most important are `host` and `dialect`, but it's good to know the others too*

### The **`sequelize`** object

When you call the `Sequelize` constructor with `new Sequelize()`, you create an instance of the `Sequelize` class. This object has several methods and attributes you can use to **interact** with your database. 

#### Attributes

- `options`: An object containing all options passed to the `Sequelize` constructor.
- `config`: The configuration object (derived from `options`) `Sequelize` uses to connect to the database.
- `models`: An object containing all initialized Models, referenced by model name.

#### Methods

*Here are some of the most common methods you'll use:*

##### `authenticate()`: 
Test the connection by trying to authenticate to the database.

```js
sequelize.authenticate() 
	.then(() => console.log('Connection has been established successfully.')) 
	.catch(error => console.error('Unable to connect to the database:', error));
```

##### `define(modelName, attributes, [options])`: 
Defines a new model, representing a table in the DB.

```js
const User = sequelize.define('User', 
	{ firstName: 
		{ type: DataTypes.STRING, 
		allowNull: false 
		}, 
	lastName: 
		{ type: DataTypes.STRING 
		// allowNull defaults to true 
	} 
});
```

##### `sync([options])`: 
*Synchronizes all defined models to the DB*

The `sync()` method is a convenience method used to synchronize the database with the models defined in your `Sequelize` instance. `sync()` creates a table if it does not exist, and does nothing if the table already exists. It works by comparing the model you've defined in your `Sequelize` instance to what's currently in your database. see [[sync vs migration]] for more insight. 

```js
sequelize.sync() 
	.then(() => console.log('All models were synchronized successfully.'));
```

The **`options`** parameter in the `sync` method is an object that could include the following properties:

- `force`: If `true`, sync will drop the table(s) first if they already exist. Default value is `false`.
    
- `alter`: If `true`, sync will update the table(s) by adding new columns, changing existing columns, or dropping columns not defined in the model. Default value is `false`.
    
- `match`: A RegExp that matches database name, for example, to only sync particular models.
    
- `schema`: The schema that the tables should be created in. This can be useful if you have multiple schemas in your PostgreSQL database.
    
- `hooks`: If `true`, sync will call Model.beforeSync, Model.sync, Model.afterSync hooks for each model being synced. Default value is `false`.

*Here's an example using `sync` with `force` set to `true`:*

```js
sequelize.sync({ force: true })
  .then(() => {
    console.log(`Database & tables created!`);
  });
```

##### `model(modelName)`: 
Fetch a Model which is already defined.

```js
const UserModel = sequelize.model('User'); 
console.log(UserModel === User); // true
```

##### `import(path)`: 
Import models defined in other files.

```js
// Import model 
// Assuming there's a file `./models/user.js` that exports a function 
// which takes a Sequelize instance and a DataTypes and returns a Model. 
const ImportedUserModel = sequelize.import('./models/user');
```

##### `transaction([options], [autoCallback])`: 
Start a transaction.

```js
// Transaction 
sequelize.transaction(t => { 
// chain all your queries here. make sure you return them. 
	return UserModel.create({ 
		firstName: 'Abraham', 
		lastName: 'Lincoln' 
	}, {transaction: t}); }) 
.then(result => { 
	console.log('Transaction has been committed'); 
}) 
.catch(error => { 
	console.error('Transaction has been rolled back', error); 
});
```

##### `query(sql, [options])`: 
Execute a query on the DB.

```js
// Execute raw SQL query 
sequelize.query("SELECT * FROM Users", { type: Sequelize.QueryTypes.SELECT}) 
	.then(results => { 
		console.log(results); 
	});
```

##### `close()`: 
Close all connections used by this `sequelize` instance.

```js
// Close connection
sequelize.close()
  .then(() => console.log('Connection closed.'))
  .catch(error => console.error('Error closing connection:', error));

```

---
## General Workflow: 

### Step 1: Install `Node.js` and `npm`

First, ensure that you have `Node.js` and `npm` installed. If you don't, you can download and install them from the [official Node.js website](https://nodejs.org/).

```
node -v
npm -v
```

### Step 2: Initialize a New `Node.js` Project

Create a new folder on your computer and navigate into it in your terminal. Once inside the folder, initialize a new `Node.js` project:

```
npm init -y
```

This will create a new `package.json` file in your folder.

### Step 3: Install Express.js and Sequelize

Next, install `Express.js` and `Sequelize` along with some additional packages such as `sequelize-cli`, `pg` and `pg-hstore` (if you're using **PostgreSQL**) with `npm`:

```
npm install express sequelize sequelize-cli pg pg-hstore
```

### Step 4: Initialize Sequelize

Initialize `Sequelize` in your project:

```
npx sequelize-cli init
```

This will create several directories: `config`, `models`, `migrations`, and `seeders`.

### Step 5: Configure Database

You need to provide your database information in the `config/config.json` file. For example:

```json
{
  "development": {
    "username": "root",
    "password": "password",
    "database": "database_name",
    "host": "127.0.0.1",
    "dialect": "postgres"
  }
}
```

*Make sure to replace `root`, `password`, and `database_name` with your actual database credentials*

### Step 6: Create a Model

Create a model using `Sequelize`. For instance, let's create a `User` model:

```
npx sequelize-cli model:generate --name User --attributes firstName:string,lastName:string,email:string
```

This will create a `user` model file in the `models` folder and a migration file in the `migrations` folder.

*Here's an example of what the `user` model file (`user.js` in the `models` directory) might look like:*

```js
'use strict';
const { Model } = require('sequelize');
module.exports = (sequelize, DataTypes) => {
  class User extends Model {
    static associate(models) {
      // Define association here
    }
  };
  User.init({
    firstName: DataTypes.STRING,
    lastName: DataTypes.STRING,
    email: DataTypes.STRING
  }, {
    sequelize,
    modelName: 'User',
  });
  return User;
};
```

The model file is where you define the shape of your data and any relationships between models. In this file, the User model is being defined with `firstName`, `lastName`, and `email` attributes, all of type `STRING`.

*And here's an example of what the migration file might look like:*

```js
'use strict';
module.exports = {
  up: async (queryInterface, DataTypes) => {
    await queryInterface.createTable('Users', {
      id: {
        allowNull: false,
        autoIncrement: true,
        primaryKey: true,
        type: DataTypes.INTEGER
      },
      firstName: {
        type: DataTypes.STRING
      },
      lastName: {
        type: DataTypes.STRING
      },
      email: {
        type: DataTypes.STRING
      },
      createdAt: {
        allowNull: false,
        type: DataTypes.DATE
      },
      updatedAt: {
        allowNull: false,
        type: DataTypes.DATE
      }
    });
  },
  down: async (queryInterface, Sequelize) => {
    await queryInterface.dropTable('Users');
  }
};
```

The migration is creating a new table called '`Users`' with the fields `firstName`, `lastName`, and `email`, along with `id`, `createdAt`, and `updatedAt`. 

### Step 7: Run Migrations

After creating the model, run the following command to execute the [[Migrations]]:

```
npx sequelize-cli db:migrate
```

### Step 8: Setting Up Express.js

Create a new file named `app.js` in your root directory and add the following code:

```js
const express = require('express');
const app = express();
const { User } = require('./models');

app.use(express.json());

app.get('/users', async (req, res) => {
  const users = await User.findAll();
  res.json(users);
});

app.post('/users', async (req, res) => {
  const { firstName, lastName, email } = req.body;
  const user = await User.create({ firstName, lastName, email });
  res.json(user);
});

app.listen(3000, () => {
  console.log('Server is running on port 3000');
});
```

Here, we've created two basic routes, a `GET` route to fetch all users and a `POST` route to create a new user. 

Before running the server, install the `nodemon` package to help automatically restart your server whenever you save a file:

```bash
npm install --save-dev nodemon
```


### Step 9: Update the `package.json`

After installing nodemon, we should add a start script to our `package.json` file so that we can easily start our server. Update the `scripts` section to look like this:

```json
"scripts": {
  "start": "nodemon app.js"
}
```

Now, you can start your server by running:

```shell
npm start
```

Your server should now be running at `http://localhost:3000`.

### Step 10: Testing Our API

You can now test your API using tools like [Postman](https://www.postman.com/) or [curl](https://curl.se/).

- For creating a new user, send a POST request to `http://localhost:3000/users` with the following JSON body:

  ```json
  {
    "firstName": "John",
    "lastName": "Doe",
    "email": "john.doe@example.com"
  }
  ```

- For getting all users, send a GET request to `http://localhost:3000/users`.

This completes the basic setup and usage of Sequelize with Express.js. Please note that this is a simple tutorial and does not include many things you would want in a production application, such as error handling and input validation.

Keep in mind that Sequelize is a powerful library that can do much more than just creating and fetching data, like managing associations between data, handling transactions, and more.

---

## Migrations with `Sequelize`

**Migrations** are like version control for your database schema. The migration feature allows you to evolve your database schema over time, while preserving existing data. It's especially helpful in production environments, where you cannot simply drop the database and recreate it from scratch.

To start using migrations, first make sure the `Sequelize` CLI is installed globally:

```bash
npm install -g sequelize-cli
```

Or as a devDependency in your project:

```bash
npm install --save-dev sequelize-cli
```

Next, initialize the project for `Sequelize`:

```bash
npx sequelize-cli init
```

This will create a new folder structure:

```
.
├── config
│   └── config.json
├── migrations
├── models
│   ├── index.js
├── seeders
```

The `migrations` directory is where your migration files will be located.

*so a migration file is almost like compilation instructions in a sense...?*

**Yes**, that's a great way to think about it.

Just like compilation instructions translate source code into executable code, a migration file translates high-level commands into **SQL** statements that modify the database schema. Each migration file provides a set of instructions for how to get your database from one state to another.

These instructions are used not only to make changes to the database, but also to track those changes over time. So if at any point you need to revert your database to an earlier state, you can do so by "undoing" the migrations, just like you can undo compilation steps.

And just like with a compilation process, where each instruction depends on the ones before it, in migrations each step builds upon the previous ones. So it's important to run your migrations in the right order, and to make sure each one completes successfully before moving on to the next. 

Also, similar to how a compiler will throw an error if it encounters an instruction it can't execute, `Sequelize` will stop the migration process if it encounters a command it can't execute, such as **trying to create a table that already exists**, or **delete a column that doesn't exist**.

### Creating a Migration

Use the following command to create a new migration:

```bash
npx sequelize-cli migration:generate --name my-migration
```

This will create a new migration file in the `migrations` directory with the name `XXXXXXXXXXXXXX-my-migration.js`, where `XXXXXXXXXXXXXX` is a timestamp.

The generated file will look like this:

```javascript
'use strict';

module.exports = {
  up: async (queryInterface, Sequelize) => {
    /*
      Add altering commands here.
      Return a promise to correctly handle asynchronicity.

      Example:
      return queryInterface.createTable('users', { id: Sequelize.INTEGER });
    */
  },

  down: async (queryInterface, Sequelize) => {
    /*
      Add reverting commands here.
      Return a promise to correctly handle asynchronicity.

      Example:
      return queryInterface.dropTable('users');
    */
  }
};
```

### Running Migrations

After you've defined what should happen in your migration, you can execute the migration with:

```bash
npx sequelize-cli db:migrate
```

If you want to undo the last migration, run:

```bash
npx sequelize-cli db:migrate:undo
```

To undo all migrations, use:

```bash
npx sequelize-cli db:migrate:undo:all
```

### Updating Migration Files

`Sequelize` keeps track of which migration files have been run by storing their names in a table in your database (by default, the table is named `SequelizeMeta`), and it won't rerun migration files that have already been executed.

Instead, the correct approach is to create a new migration that will adjust your database schema to align with the changes in your model.

*For example*, let's say you have a `User` model and you've already created a migration for it. Later, you decide to add a new `username` field to the `User` model. You shouldn't edit the existing migration file. Instead, you should create a new migration file that adds the `username` field to the `Users` table in your database:

```bash
npx sequelize-cli migration:generate --name add-username-to-user
```

*Then*, in the generated migration file, you would write something like this:

```javascript
'use strict';

module.exports = {
  up: async (queryInterface, Sequelize) => {
    await queryInterface.addColumn('Users', 'username', {
      type: Sequelize.STRING,
      allowNull: false,
      unique: true,
    });
  },

  down: async (queryInterface, Sequelize) => {
    await queryInterface.removeColumn('Users', 'username');
  }
};
```

*Then*, you would run your migrations again:

```bash
npx sequelize-cli db:migrate
```

This would add the `username` column to the `Users` table in your database.

This approach — **creating a new migration for each change to your database schema** — allows you to keep track of the changes over time, and it allows you to roll back specific changes if needed. It's similar to how you might use version control like git for your code.

### Best Practices

- Always use migrations for schema changes, not manual SQL or Sequelize's sync method. This allows you to easily track and replicate schema changes across multiple environments.
- Always use `async/await` in your migration functions to correctly handle asynchronicity.
- Include both `up` and `down` methods in your migrations to allow for rolling back changes if necessary.
- Be mindful of the order of your migrations, as they are run in the order they were created.
- When altering a table, consider potential effects on existing data. For example, if you make a column non-nullable, ensure existing rows don't contain null in that column.
- Test your migrations before running them on production data.

Remember, migrations are a powerful tool for managing database schemas, but they should be used carefully and tested thoroughly to ensure data integrity.


`Sequelize` does **not** have built-in functionality to automatically generate migrations based on changes made to models. You'll have to manually generate and write migrations whenever you make changes to your models that require a change in the database schema.

For example, if you add a new field to your model, you'll need to create a new migration that adds a column for that field in the relevant database table.

**Here is how you might do it:**

### 1. Generate a new migration with the Sequelize CLI:

```bash
npx sequelize-cli migration:generate --name add-field-to-model
```

*Note:* each time you run the `model:generate` command with the Sequelize CLI, it will create a new model file in the `models` directory. It doesn't add the new model to an existing model file.

So, if you run the command twice with two different table names, you'll end up with two separate files in the `models` directory. For example:
```bash
npx sequelize-cli model:generate --name User --attributes firstName:string,lastName:string,email:string
npx sequelize-cli model:generate --name Product --attributes name:string,price:integer
```

This will generate a `user.js` file **and** a `product.js` file in the `models` directory, each containing the respective model definition.

### 2. In the generated migration file, write the code in the `up` function to add the new column:

```javascript
up: async (queryInterface, Sequelize) => {
  await queryInterface.addColumn('TableName', 'fieldName', Sequelize.STRING);
}
```

### 3. Similarly, write the code in the `down` function to undo the migration:

```javascript
down: async (queryInterface, Sequelize) => {
  await queryInterface.removeColumn('TableName', 'fieldName');
}
```

### 4. Run the migration:

```bash
npx sequelize-cli db:migrate
```

There are tools and libraries available (like `sequelize-auto-migrations` [docs](https://www.npmjs.com/package/sequelize-auto-migrations)) that can automate the process of creating migrations based on model changes, but these are not part of `Sequelize` itself and might not be suitable or stable for all use-cases. Always review and test any automated migrations thoroughly before applying them to a production database.

---
## Associations: 

In `Sequelize`, there are three types of associations you can use: **One-To-One**, **One-To-Many**, and **Many-To-Many**.

### One-To-One Associations

In a **one-to-one** association, an instance of a model is associated with exactly one instance of another model. For example, suppose we have `User` and `Profile` models. Each `User` can have exactly one `Profile`, and each `Profile` belongs to exactly one `User`.

We can establish this association using the `hasOne` and `belongsTo` methods:

```javascript
const User = sequelize.define('User', { /* ... */ });
const Profile = sequelize.define('Profile', { /* ... */ });

User.hasOne(Profile);
Profile.belongsTo(User);
```
*Do we need the `belongsTo()` method if we've already declared the `hasOne()` method?*

**Yes**, you should define both sides of the association. This is necessary to set up foreign key relationships correctly and to enable `Sequelize` to be able to make queries and join statements in either direction.

-   `User.hasOne(Profile)`: This adds a foreign key on the `Profile` model, which points to the `User` model. It allows you to do `User.getProfile()`.

-   `Profile.belongsTo(User)`: This allows you to do `Profile.getUser()`.


If you only define `User.hasOne(Profile)`, but not `Profile.belongsTo(User)`, you will not be able to access the `User` associated with a `Profile` instance.

Similarly, if you only define `Profile.belongsTo(User)`, but not `User.hasOne(Profile)`, you will not be able to access the `Profile` associated with a `User` instance.

Defining associations in both directions allows Sequelize to set up all the methods and hooks necessary for creating, retrieving, updating and deleting associated records correctly.

### One-To-Many Associations

In a **one-to-many** association, one instance of a model can be associated with multiple instances of another model. For example, suppose we have `User` and `Post` models. Each `User` can have multiple `Posts`, but each `Post` belongs to exactly one `User`.

We can establish this association using the `hasMany` and `belongsTo` methods:

```javascript
const User = sequelize.define('User', { /* ... */ });
const Post = sequelize.define('Post', { /* ... */ });

User.hasMany(Post);
Post.belongsTo(User);
```

### Many-To-Many Associations

In a **many-to-many** association, multiple instances of a model can be associated with multiple instances of another model. For example, suppose we have `User` and `Project` models. Each `User` can be associated with multiple `Projects`, and each `Project` can be associated with multiple `Users`.

We can establish this association using the `belongsToMany` method:

```javascript
const User = sequelize.define('User', { /* ... */ });
const Project = sequelize.define('Project', { /* ... */ });

User.belongsToMany(Project, { through: 'UserProject' });
Project.belongsToMany(User, { through: 'UserProject' });
```

Here, 'UserProject' is the name of the join table, which Sequelize will create for you.

### Using Associations in Express

In your Express routes, you can use Sequelize's `include` option to include the associated data in your queries:

```javascript
app.get('/users', async (req, res) => {
  const users = await User.findAll({ include: [Profile] });
  res.json(users);
});
```

This will return each `User` along with their `Profile`.

If you want to include nested associations, you can do so like this:

```javascript
app.get('/users', async (req, res) => {
  const users = await User.findAll({
    include: {
      model: Profile,
      include: [AnotherModel]
    }
  });
  res.json(users);
});
```

This will return each `User` along with their `Profile` and the `Profile's` `AnotherModel`.

Remember that you need to first define these associations in your models before you can include them in your queries.

Also, if you're creating, updating, or deleting associated data, you may need to use Sequelize's `create`, `update`, and `destroy` methods along with the association methods, such as `add`, `set`, `remove`, etc., which Sequelize provides for manipulating associated data.


---
## Validations: 

Validations and constraints are a vital part of working with data in Sequelize and Express.js. They help to maintain data integrity by ensuring only valid data is saved in your database.

**Validations in Sequelize**

Validations in Sequelize are specified in the model definition and provide a variety of ways to validate data before it gets saved into the database.

*Here is an example of how you can define a model with some validations:*

```javascript
const User = sequelize.define('User', {
  email: {
    type: Sequelize.STRING,
    allowNull: false,
    unique: true,
    validate: {
      isEmail: true,
    },
  },
  password: {
    type: Sequelize.STRING,
    allowNull: false,
    validate: {
      len: [8, 20],
    },
  },
});
```

In the above example, we have a `User` model with two fields: `email` and `password`.

For the `email` field:

- `allowNull: false` means that this field cannot be null.
- `unique: true` means that this field must be unique in the database. 
- `validate: { isEmail: true }` means that this field must be a valid email.

For the `password` field:

- `allowNull: false` means that this field cannot be null.
- `validate: { len: [8, 20] }` means that the password length must be between 8 and 20 characters.

Sequelize provides a variety of validation options like `is`, `isUrl`, `isIP`, `isAlpha`, `isAlphanumeric`, `isNumeric`, `isInt`, `isLowercase`, `isUppercase`, `notNull`, `isNull`, `notEmpty`, and so on.

**Handling Validation Errors in Express**

When you try to save a model instance using Sequelize, it will automatically run the validations. If any validation fails, Sequelize will throw a `SequelizeValidationError`.

You can catch this error in your Express route handlers and return a response with the validation error messages. Here is an example:

```javascript
app.post('/users', async (req, res) => {
  try {
    const user = await User.create(req.body);
    res.json(user);
  } catch (err) {
    if (err instanceof Sequelize.ValidationError) {
      return res.status(400).json({ errors: err.errors.map(e => e.message) });
    }
    return res.status(500).json({ error: 'Something went wrong' });
  }
});
```

In the above example, if the `User.create()` call fails due to a validation error, we catch the error, check if it's a `SequelizeValidationError`, and if it is, we return a 400 response with the validation error messages. If it's not a `SequelizeValidationError`, we return a 500 response.

**Constraints in Sequelize**

In addition to validations, you can also define constraints in Sequelize. **Constraints** are rules enforced by the database. If a constraint is violated, the database will reject the operation.

Constraints in Sequelize are defined as options in the model definition or column definition. Some common constraints in Sequelize include `primaryKey`, `allowNull`, `unique`, `references` (for foreign keys), etc.

For example, here is how you can define a model with a primary key and a foreign key:

```javascript
const User = sequelize.define('User', {
  id: {
    type: Sequelize.INTEGER,
    primaryKey: true,
    autoIncrement: true,
  },
  teamId: {
    type: Sequelize.INTEGER,
    references: {
      model: 'teams',
      key: 'id',
    },
  },
});
```


---
## Password Management: 
here's how you might use `bcrypt` to hash passwords before storing them. You'd also check the hashed password when the user logs in.

First install `bcrypt` with:

```bash
npm install bcrypt
```

Then modify the Express app like so:

```javascript
import bcrypt from 'bcrypt'

...

const User = sequelize.define('User', {
  email: {
    type: DataTypes.STRING,
    allowNull: false,
    unique: true,
    validate: {
      isEmail: true
    }
  },
  password: {
    type: DataTypes.STRING,
    allowNull: false,
    validate: {
      len: [6, 50]
    }
  }
});

// ... rest of your models and associations

// Before a User is created, automatically hash their password
User.beforeCreate(async (user, options) => {
  const salt = await bcrypt.genSalt(10);
  user.password = await bcrypt.hash(user.password, salt);
});
```

This code uses `bcrypt` to automatically hash a user's password **before** it is stored in the database. When a user tries to log in, it compares the hashed password in the database with the password provided by the user.

Note: In a real-world application, you wouldn't just send the user's information back to the client after a successful login. You'd typically generate a token (like a [[JWT (JSON Web Token)]]) that the client can use for authentication in future requests.

---
## Full CRUD Example With Express

Here's an example of an Express application that uses `Sequelize` to manage a **many-to-many** relationship with a bridge table, includes validation, and provides **REST** operations. This example consists of `User` and `Course` models, with a `UserCourse` model serving as the bridge table.

**UML**

```lua
+-------+       +-------------+       +--------+
| User  |       | UserCourse  |       | Course |
+-------+       +-------------+       +--------+
| id(PK)| <---> | userId      |       | id(PK) |
| email |       | courseId <---> | id |
| pass  |       | status      |       | name   |
+-------+       +-------------+       +--------+

```

**Code**

```javascript
// Import required modules
const express = require('express');
const { Sequelize, DataTypes } = require('sequelize');

// Initialize Express
const app = express();

// Initialize Sequelize
const sequelize = new Sequelize('database', 'username', 'password', {
  host: 'localhost',
  dialect: 'mysql',
});

// Define the User model
const User = sequelize.define('User', {
  email: {
    type: DataTypes.STRING,
    allowNull: false,
    unique: true,
    validate: {
      isEmail: true
    }
  },
  password: {
    type: DataTypes.STRING,
    allowNull: false,
    validate: {
      len: [6, 50]
    }
  }
});

// Before a User is created, automatically hash their password
User.beforeCreate(async (user, options) => {
  const salt = await bcrypt.genSalt(10);
  user.password = await bcrypt.hash(user.password, salt);
});

// Define the Course model
const Course = sequelize.define('Course', {
  name: {
    type: DataTypes.STRING,
    allowNull: false,
  },
});

// Define the UserCourse model (bridge table)
const UserCourse = sequelize.define('UserCourse', {
  status: {
    type: DataTypes.STRING,
    allowNull: false,
    validate: {
      isIn: [['enrolled', 'completed']]
    }
  },
});

// Setup associations
User.belongsToMany(Course, { through: UserCourse });
Course.belongsToMany(User, { through: UserCourse });

// Sync the database
sequelize.sync();

// Parse JSON request body
app.use(express.json());

// REST operations
app.post('/users', async (req, res) => {
  const { email, password } = req.body;
  const newUser = await User.create({ email, password });
  res.json(newUser);
});

app.get('/users/:id', async (req, res) => {
  const user = await User.findByPk(req.params.id);
  res.json(user);
});

app.post('/courses', async (req, res) => {
  const { name } = req.body;
  const newCourse = await Course.create({ name });
  res.json(newCourse);
});

app.post('/enroll', async (req, res) => {
  const { userId, courseId, status } = req.body;
  const enrollment = await UserCourse.create({ userId, courseId, status });
  res.json(enrollment);
});

// Start server
app.listen(3000, () => console.log('Server is running on port 3000'));
```

*Note*, in most real-world applications, it is best practice to keep each model in its own JavaScript file. This enhances the organization of your project and makes it easier to manage as your application grows.

*Remember* to replace `'database'`, `'username'`, and `'password'` with your actual database name, username, and password.
