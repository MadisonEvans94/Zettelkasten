#evergreen1
###### upstream: [[Web Development]]

## Introduction: 

**`Express.js`** is a minimalistic and flexible web application framework for [[Node.js]], a runtime environment that executes JavaScript code outside of a web browser. It simplifies the process of building web applications and APIs by providing a set of robust features and tools.

`Express.js` is designed to be un-opinionated, allowing developers to structure their applications in various ways and giving them the freedom to choose the components they want to use. It acts as a lightweight layer on top of `Node.js`, providing a simple and intuitive API for handling HTTP requests and responses.

---

## Quick Start Workflow: 

*jump to `full boiler plate` section at bottom of **Quick Start Workflow** for full boilerplate*

*see [[Creating a CLI (Command Line Interface) script to automate the Express server setup]] for instructions on how to automate this step* 

### 1. Creating the `app` file

#### 1. First, make sure you are in a new project directory, and run ...

best to do this inside a directory called `server` for good organization

```bash
npm init -y
```

... to create a new `package.json` file. Then, install express with ...

```bash
npm install express
```

#### 2. Then, create an `app.js` file with the following code: 

*see [[app.listen()]] for more details about how the method works*

```javascript 
//imports
const express = require("express");
const process = require("process");

//create the app object
const app = express();

//MIDDLEWARE

//DEFINE PORT
const port = process.env.PORT || 5001;

//start the app listening on desired port 
app.listen(port, () => {
  console.log(`Server is listening at http://localhost:${port}`);
});

//handle GET request 
app.get('/', (req, res) => {
  res.send('Hello, World!');
});

```
*see [[Attributes and Methods of the App Object]] for more...* 

#### 3. To run the server, run the following from the command line: 
```bash 
node app.js
```

### 2. Using `nodemon` for automatic server updates: 

`nodemon` is a utility that monitors for any changes in your source files and automatically restarts your server. It's perfect for development, as you don't have to manually stop and start your server every time you make a change. [docs](https://www.npmjs.com/package/nodemon)

Here's how to install and use `nodemon`:

#### Step 1: Install `nodemon`

First, you'll need to install `nodemon` as a development dependency in your project. This can be done using `npm`. You should run this command in your terminal in the project directory:

```bash
npm install --save-dev nodemon
```

The `--save-dev` flag is used to add the package to your `devDependencies` list, meaning it's only required for development, not for running the application in production.

#### Step 2: Update your package.json file

After installing `nodemon`, you'll need to modify your `package.json` file to use `nodemon` to start your server instead of `node`. In the `scripts` section of your `package.json` file, change the `start` script to use `nodemon`:

```json
"scripts": {
    "start": "nodemon app.js"
}
```

#### Step 3: Run your server with nodemon

Now you can start your server with `nodemon` by using the `npm start` command in your terminal:

```bash
npm start
```

This will start your server and `nodemon` will watch for changes. Every time you save a file, `nodemon` will automatically restart your server.

Note: The file that nodemon will watch is the `app.js` in this case, but this can be replaced with whatever your server's entry point file is.

#### Summary:

- Install `nodemon` as a development dependency using `npm`
- Update `package.json` to start the server with `nodemon`
- Use `npm start` to start the server with `nodemon`
- `nodemon` will now restart the server every time you save changes

### 3. Diagnostic Logging setup with `morgan`: 

`Morgan` is a popular HTTP request logger middleware for Node.js applications. It simplifies the process of logging requests to your application, which can be helpful for debugging and understanding your application's traffic.

#### Step 1: Install Morgan

First, you'll need to install `Morgan` as a dependency in your project. You can do this with npm. Run this command in your terminal in your project directory:

```bash
npm install morgan
```

#### Step 2: Import Morgan in your app

After installing `Morgan`, you need to import it into your Express app. You can do this with an `import` statement at the top of your `app.js` file:

```javascript
const morgan = require('morgan');
```

#### Step 3: Use Morgan Middleware

Next, you need to tell your Express app to use `Morgan` as a middleware. Add this line of code **after** creating your app, but **before** defining your routes:

```javascript
app.use(morgan('dev'));
```

The `'dev'` argument tells `Morgan` to log messages in a concise, colored 'development-friendly' format. You can replace `'dev'` with other string options like `'tiny'`, `'short'`, `'combined'`, etc. to change the format of the logs.

Now, `Morgan` will automatically log HTTP requests to your console. The logs will include details like the HTTP method, the request endpoint, the HTTP status code, response time, and more.

#### Step 4: Run your server

Now you can start your server like normal:

```bash
npm start
```

You should now see `Morgan` logging each incoming request in your console.

**Example**: 
```bash
GET / 200 3.530 ms - 26
```

#### Summary:

- Install `Morgan` as a dependency using npm
- Import `Morgan` in your Express app
- Use `Morgan` as a middleware in your Express app
- Start your server and see `Morgan` log incoming requests

### 4. Enable Middleware

See the **Middleware** section for more


### 5. Set up CRUD Functionality for DB

this step will vary depending on the type of database you're using. see the **`Sequelize` ORM** section for more

### Full Boiler Plate: 

```bash 
npm i sqlite3 cors helmet morgan express sequelize 
```

```js
//imports
const express = require("express");
const process = require("process");
const morgan = require("morgan");
const cors = require("cors");
const helmet = require("helmet");

//db object import for models
const db = require("./models");

//initialize app object
const app = express();

//DEFINE PORT
const port = process.env.PORT || 5001;

//MIDDLEWARE
app.use(helmet());
app.use(cors());
app.use(morgan("dev"));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

//LISTEN ON PORT
app.listen(port, () => {
	console.log(`now listening on port ${port}`);
});

//ROUTES
app.get("/", (req, res) => {
	res.send("Welcome to the app!");
});

app.get("/api/", (req, res) => {
	res.send("Welcome to the api");
});

app.post("/api/student", (req, res) => {
	db.Student.create({
		name: req.body.name,
	})
		.then((student) => {
			res.status(201).json(student);
		})
		.catch((error) => {
			res
				.status(500)
				.json({ error: "There was an error creating the student." });
		});
});

app.get("/api/student", (req, res) => {
	db.Student.findAll()
		.then((students) => {
			res.status(200).json(students);
		})
		.catch((error) => {
			res
				.status(500)
				.json({ error: "There was an error retrieving students." });
		});
});

app.get("/api/student/:id", (req, res) => {
	const id = req.params.id;
	db.Student.findByPk(id)
		.then((student) => {
			if (student) {
				res.status(200).json(student);
			} else {
				res.status(404).json({ error: "Student not found." });
			}
		})
		.catch((error) => {
			res
				.status(500)
				.json({ error: "There was an error retrieving the student." });
		});
});
```

---

## Middleware: 
*see more at [[Express Middleware]]*

---

## The `req` , `res`, and `next` arguments: 

### `req` 
represents the HTTP request from the client, containing all request data.

### `res` 
is used to configure and send the HTTP response from the server back to the client.

### `next` 
a function to pass control to the next middleware in the stack, allowing for efficient handling of requests and responses in the HTTP cycle.

*see [[req, res, and next in depth]] for more*

---

## Important `app` Methods: 

### **`app.use()`**: 

This function is used to mount **middleware** functions to the app. It takes 3 arguments: `req`, `res`, and `next` [docs](https://expressjs.com/en/guide/using-middleware.html)

#### Custom Middleware Example
```javascript
app.use((req, res, next) => {
  res.send(req.params.id)
  next()
})
```

#### 3rd Party Middleware Example
```js
app.use(express.json()); // This middleware parses incoming requests with JSON payloads
```

### **`app.get()`**: 

This method is used to handle `HTTP GET` requests at a specific route. It takes two parameters: the path and the callback function which is executed when the route is hit. 

```js
app.get('/', (req, res) => {
  res.send('Hello World!');
});
```

*^In this example, when a client sends a `GET` request to the root (`/`) of the application, `'Hello World!'` is sent as a response.* (see [[the difference between .get() and .use()]] for more details)*

### **`app.post()`**: 

Similar to `app.get()`, but this is used to handle `HTTP POST` requests. It is usually used to create new data.
```js
app.post('/users', (req, res) => {
  // Code to create a new user
  // Assuming newUser is the created user
  res.status(201).json(newUser);
});
```

*^Here, when a client sends a POST request to `/users`, a new user is created, and that new user is sent back to the client.*

### **`app.put()`**: 

This is used to handle `HTTP PUT` requests. It is typically used to update existing data.

```js
app.put('/users/:id', (req, res) => {
  // Code to update a user by req.params.id
  // Assuming updatedUser is the updated user
  res.json(updatedUser);
});
```

*^In this case, when a client sends a PUT request to `/users/{id}`, the user with that ID is updated.*

### **`app.delete()`**: 

This is used to handle `HTTP DELETE` requests. It is used to delete existing data.

```js
app.delete('/users/:id', (req, res) => {
  // Code to delete a user by req.params.id
  res.status(204).send();
});
```

*See [[REST API example with Express]] for more details*

---

## Server to Server Requests in Express: 

The `axios` library and `fetch` api functions the same way, regardless of whether it's used in a **server-side** `Node.js` (such as `Express.js`) application, or in a **client-side** library/framework like `React.js`.

The main difference comes in the context and what you do with the data you receive:

- **On the server side (`Express.js`)**, you might use `Axios` to fetch some data from another server, manipulate that data, and then send it along to your frontend. You might also use it to aggregate data from multiple sources, caching, or handle server-to-server communication.

- **On the client side (`React.js`)**, you would typically use `Axios` to get data from your own server (or some third-party API) and then use that data to update the state of your components and render them.

---


## `Sequelize` ORM: 

[Sequelize Models - Storing Backend Data](https://www.youtube.com/watch?v=ikJ5AXDj3go&ab_channel=ChrisCourses)

**`Sequelize`** is a popular [[Object Relational Mapping (ORM)]] library for SQL databases in Express applications. It supports multiple SQL dialects including PostgreSQL, MySQL, SQLite, and MSSQL. `Sequelize` provides a host of features including transaction support, relations, read replication, and more. [docs](https://sequelize.org/)

### Workflow 

#### Install Dependencies 
```bash
npm install sequelize 
```

`Sequelize` also provides a utility to automatically create **migration** files and apply them. 

To do that, install the **`Sequelize CLI`**:

```bash
npm install --save-dev sequelize-cli
```

...then, initialize your project for `Sequelize`:

```bash
npx sequelize-cli init
```

This command will create a basic project structure with the following directories: `config`, `models`, `migrations`, and `seeders`.

#### Update `config.json`
...by adding this to development object: 

```json
"storage": "./database.sqlite"
```


##### 1. update `models/index.js` as follows: 

*be sure to download `sqlite3`*

```js
const fs = require("fs");
const path = require("path");
const Sequelize = require("sequelize");

const basename = path.basename(__filename);
const db = {};

const sequelize = new Sequelize({
	dialect: "sqlite",
	storage: "./database.sqlite",
});

const modelFiles = fs.readdirSync(__dirname).filter((file) => {
	return (
		file.indexOf(".") !== 0 &&
		file !== basename &&
		file.slice(-3) === ".js" &&
		file.indexOf(".test.js") === -1
	);
});

Promise.all(
	modelFiles.map((file) => {
		const model = require(path.join(__dirname, file));
		const modelName = model.name;
		db[modelName] = model(sequelize, Sequelize.DataTypes);
	})
).then(() => {
	Object.keys(db).forEach((modelName) => {
		if (db[modelName].associate) {
			db[modelName].associate(db);
		}
	});
	db.sequelize = sequelize;
	db.Sequelize = Sequelize;
});

module.exports = db;
```

##### 2. Import `db` object into `app.js`:

```js
const express = require("express");
const process = require("process");
const morgan = require("morgan");
const cors = require("cors");

  
//DB OBJECT IMPORT 
const db = require("./models");
```

*the `db` object is an object that will contain all your models as properties, as well as the `Sequelize` and `sequelize` instances. Here's what it may look like:*

```js
db = {
  User: User,
  // other models you have defined
  sequelize: sequelize,
  Sequelize: Sequelize
};

```

whenever you want to interact with the database through the express app, such as performing **[[CRUD]]** operations, you should do so through the **`db`** object and its methods. 
*See [[Understanding the db object in models directory]] for more details*

#### Define Models

**use `cli`**
```bash
npx sequelize-cli model:generate --name User --attributes name:string
```

**...or just create a file in the `./models` directory**
```js
//models/user.js

const { DataTypes } = require("sequelize");

const User = (sequelize, DataTypes) => {
	return sequelize.define("User", {
		name: DataTypes.STRING,
	});
};

module.exports = User;
```

The `User` function is essentially a [[Factory]] function for creating a `User` model given a `sequelize` instance and its `DataTypes`. This all follows the [[Module Pattern]]. 


*See [[Guide for Connecting Sequelize to DB Engine]] for details on connecting to MySQL, Postgre, etc...*

*Also note...* Another commonly used library is **Mongoose**, which is an [[Object Data Modeling (ODM)]] library for MongoDB and `Node.js`. It helps to manage relationships between data, provides schema validation, and is used to translate between objects in code and the representation of those objects in MongoDB.

### Using `Sqlite` for development: 

To use SQLite in your `Node.js` application with `Sequelize`, you will need to install the `sqlite3` module using `npm` or yarn. Here is how you can do it:

```bash
npm install sqlite3
```

In your `models/index.js` file, you would have to update the configuration section. Instead of configuring a connection to a typical SQL database server, you can configure it to connect to an SQLite file. 

Be sure to have this line in your `models/index.js` file:

```javascript
const sequelize = new Sequelize({
	dialect: "sqlite",
	storage: "./database.sqlite",
});
```

With the above configuration, `Sequelize` will connect to an SQLite database in your project directory named `database.sqlite`.

### Migrations with `sequelize`
*Note*, for sqlite, you need to use **CommonJS**. See [[Modules vs CommonJS]] for details

*Let's go through the workflow...*
```bash
npx sequelize-cli migration:generate --name create-user
```

#### Creating a Table

```js
'use strict';
/** @type {import('sequelize-cli').Migration} */

module.exports = {
	async up (queryInterface, Sequelize) {
		/**		
		* Add altering commands here.		
		*		
		* Example:		
		* await queryInterface.createTable('users', { id: Sequelize.INTEGER });		
		*/
},
	
	async down (queryInterface, Sequelize) {
		/**
		* Add reverting commands here.
		*
		* Example:
		* await queryInterface.dropTable('users');
		*/
	}
};
```

```js
'use strict';
/** @type {import('sequelize-cli').Migration} */

module.exports = {
	async up(queryInterface, Sequelize) {
		return queryInterface.createTable("Students", {
			id: {
				allowNull: false,
				autoIncrement: true,
				primaryKey: true,
				type: Sequelize.INTEGER,
			},
			name: {
				type: Sequelize.STRING,
			},
			createdAt: {
				allowNull: false,
				type: Sequelize.DATE,
			},
			updatedAt: {
				allowNull: false,
				type: Sequelize.DATE,
			},
		});
	},
	
	async down(queryInterface, Sequelize) {
		return queryInterface.dropTable("Students");
	},
};

```

*Here's what this migration does:*

- The `up` method creates the `Students` table with four columns:
    - `id`: A unique identifier for each record. It auto-increments and is the primary key.
    - `name`: A string column for the student name.
    - `createdAt`: A timestamp for when the record was created.
    - `updatedAt`: A timestamp for when the record was last updated.
- The `down` method drops the `Students` table. It's used to undo the changes made by the `up` method.

If you run `npx sequelize-cli db:migrate` with your current SQLite configuration and after updating the migration file as shown above, `Sequelize` will execute the `up` method in your migration, which will create the `Students` table in your SQLite database.

If the `Students` table is successfully created, `Sequelize` will log this event in a table called `SequelizeMeta` in your SQLite database. This table keeps track of which migrations have been run, to prevent them from being run multiple times. You don't need to interact with this table directly, `Sequelize` will handle it for you.

*see [[Testing with Postman]] for more*
#### Running Migrations

Running migrations is also done using the `sequelize-cli` tool. Once you've set up your migration, you can run it with the following command:

```bash
npx sequelize-cli db:migrate
```

This command will run all pending migrations, in the order they were created.

#### Undoing Migrations

If you need to **undo the last migration** that you ran, you can do so with the following command:

```bash
npx sequelize-cli db:migrate:undo
```

If you need to **undo all migrations**, you can use the following command:

```bash
npx sequelize-cli db:migrate:undo:all
```

Remember that the `down` function in your migration file is what will be executed when you undo a migration, so make sure to implement it properly.

### Seeding Your Database via Seeders Directory

**blog**: [Seeding data with Sequelize](https://dev.to/idmega2000/seeding-data-with-sequelize-1f3o)

Database seeding is the initial seeding of a database with data. This data can be **dummy data** used for testing or real data used to initialize the application. `Sequelize` provides a powerful system for managing data migrations and seed data. 


#### Creating a Seeder

To generate a seeder, you can use the `seed:generate` command:

```bash
npx sequelize-cli seed:generate --name demo-user
```

This command creates a new file in the `seeders` directory. The name of the file will include a timestamp and the name you specified (in this case, "demo-user").

Here is what the generated file might look like:

```javascript
'use strict';

module.exports = {
  up: async (queryInterface, Sequelize) => {
    /*
      Add altering commands here.
      Return a promise to correctly handle asynchronicity.

      Example:
      return queryInterface.bulkInsert('People', [{
        name: 'John Doe',
        isBetaMember: false
      }], {});
    */
  },

  down: async (queryInterface, Sequelize) => {
    /*
      Add reverting commands here.
      Return a promise to correctly handle asynchronicity.

      Example:
      return queryInterface.bulkDelete('People', null, {});
    */
  }
};
```

In this file, you will define what data you want to insert into the database.

#### Defining Seed Data

Let's say we have a `Users` table and we want to insert some demo users. We can define this data in the `up` method:

```javascript
module.exports = {
  up: async (queryInterface, Sequelize) => {
    return queryInterface.bulkInsert('Users', [{
      username: 'demo',
      password: 'demo',
      email: 'demo@example.com',
      createdAt: new Date(),
      updatedAt: new Date()
    }], {});
  },

  down: async (queryInterface, Sequelize) => {
    return queryInterface.bulkDelete('Users', null, {});
  }
};
```

Here, we're using the `bulkInsert` method to insert a new user into the `Users` table. The `down` method does the opposite: it deletes the user from the `Users` table. This is useful if you ever need to rollback your seed data.

#### Running Seeders

Once you've defined your seed data, you can run it using the `db:seed:all` command:

```bash
npx sequelize-cli db:seed:all
```

This runs all of the seeders in the `seeders` directory.

If you need to undo the seed data, you can use the `db:seed:undo:all` command:

```bash
npx sequelize-cli db:seed:undo:all
```

This undoes all of the seed data, using the `down` method in each seeder file.

#### Best Practices

- Always use the `bulkInsert` method when inserting multiple records, as it is much faster than inserting records one at a time.
- Always include a `down` method in your seeders to undo the seeding. This can be very useful during testing and development.
- Try to keep your seed data simple and not tied to specific business logic. It should be easy to understand and easy to modify.
---



### Testing **CRUD** Functionality in Command Line: 

Let's assume that `app.js` looks like this 

```js

const app = express();
app.use(express.json());

User(sequelize, Sequelize.DataTypes);

app.listen(3000, () => {
    console.log('App is running on port 3000');
});

```

#### Create (POST)

To **create** a new user from the command line, you can use curl. Here's an example:
```bash
curl -X POST -H "Content-Type: application/json" -d '{"username":"John", "birthday":"2000-01-01"}' http://localhost:3000/users
```

#### Read (GET)

To **read** a list of users, use the following command:
```bash
curl -X GET http://localhost:3000/users
```

To **read** a specific user by ID:
```bash
curl -X GET http://localhost:3000/users/:id
```

*Replace `:id` with the actual user ID*

#### Update (PUT)

To **update** a user, use the following command:
```bash
curl -X PUT -H "Content-Type: application/json" -d '{"username":"JohnUpdated", "birthday":"2000-02-02"}' http://localhost:3000/users/:id
```

#### Delete (DELETE)

To **delete** the user, use the following command: 
```bash
curl -X DELETE http://localhost:3000/users/:id
```

To create a new user from the command line, you can use curl. Here's an example:

## Routing

**TODO**

---

## AWS Integration: 

This section will guide you through the process of setting up an **`Express.js`** server on AWS through *2* options: **EC2** and **Elastic Beanstalk**

### AWS EC2

Amazon Elastic Compute Cloud ([[EC2]]) is a part of Amazon's cloud-computing platform, **Amazon Web Services (AWS)**, that allows users to rent virtual computers to run their own computer applications.

*See [[Launching an Express app on EC2]] for more instructional detail*

### AWS Elastic Beanstalk

AWS Elastic Beanstalk is a service for deploying and scaling web applications and services developed in JavaScript, .NET, PHP, `Node.js`, Python, Ruby, Go, and Docker.

*See [[Launching an Express app on Elastic Beanstalk]] for more instructional detail*

---
## Best Practice Setup: 
*when writing code for an express server, what is the best practice way of setting up all the settings and middleware for your app?*

Setting up the settings and middleware for your Express app can vary depending on the specific needs of your project. However, there are some general best practices you can follow:

### 1. Organization and Structuring:

-   Use a modular structure. Your app's directory could be divided into a few main folders like `routes`, `models`, `controllers`, `middleware`, and `public`. This organization helps with maintaining and understanding the code.
-   Follow the [[Model View Controller (MVC)]] pattern if it fits your application. This further abstracts the routes, business logic, and data models in your application.

### 2. Middleware and Settings Configuration:

-   Middleware functions that apply globally to the app should be placed near the top of your server file or in your main app configuration file, right after where you instantiate your app with `const app = express()`. This ensures that these functions are loaded and run before any routes or endpoints are defined.
-   Always put the `express.json()` middleware near the top of your middleware stack if you're expecting to receive JSON data in your requests.
-   If using third-party middleware, make sure to install and import them correctly, and read through their documentation to understand their specific setup process.
-   Organize your routes in separate files, and use the `express.Router` class to encapsulate route handling for different parts of your app. Then, you can use the `app.use()` function in your main server file to set up the paths for these routers.

*Here is a basic example:*

```js
//imports
const express = require("express");
const process = require("process");
const morgan = require("morgan");
const cors = require("cors");
const helmet = require("helmet");

//db object import for models
const db = require("./models");

//initialize app object
const app = express();

//DEFINE PORT
const port = process.env.PORT || 5001;

//LISTEN ON PORT
app.listen(port, () => {
	console.log(`now listening on port ${port}`);
});

//MIDDLEWARE
app.use(helmet());  // set various HTTP headers to help secure your app
app.use(cors());  // enable CORS
app.use(morgan('dev'));  // log HTTP requests
app.use(express.json());  // parse incoming request bodies
app.use(express.urlencoded({ extended: true }));  // parse incoming request bodies

const port = process.env.PORT || 3000;
app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});
```

### 3. Environment Variables:

-   Use environment variables for sensitive data like your database connection URI, your session secret, and your app's port if it's not deployed on a standard port.
-   Use a package like `dotenv` to load environment variables from a `.env` file into `process.env`.

### 4. Error Handling:

-   Always add error-handling middleware at the end of your middleware stack. For example, this error-handling middleware function adds HTTP headers for an error and handles any error that occurs in the app:
```js
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).send('Something broke!');
});
```


