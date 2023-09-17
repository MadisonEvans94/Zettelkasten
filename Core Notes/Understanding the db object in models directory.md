#incubator 
upstream:

---

**video links**: 

---

## The `db` Object

The `db` object is a simple JavaScript object that has a property for each of your models. If you have models named `User` and `Course`, you would access them as `db.User` and `db.Course`. These methods all return promises, because they perform **asynchronous** operations that communicate with the database.

### Models 

Each model is a **function** that can be used to create new instances of that model, and has a host of static methods for querying the database, such as `findAll`, `findOne`, `create`, `update`, and `destroy`. 

#### `findAll`

Used to find all instances in the database for a certain model. It's useful when you want to retrieve all records from a table.

```javascript
const users = await db.User.findAll();
console.log(users);
```

#### `findOne`

Used to find a specific instance in the database. It's useful when you want to retrieve a specific record from a table based on some condition.

```javascript
const user = await db.User.findOne({ where: { username: 'johndoe' } });
console.log(user);
```

#### `create`

Used to create a new instance in the database. It's useful when you want to insert a new record into a table.

```javascript
const newUser = await db.User.create({ username: 'johndoe', email: 'johndoe@gmail.com' });
console.log(newUser);
```

#### `update`

Used to update an instance in the database. It's useful when you want to modify an existing record in a table.

```javascript
const [updated] = await db.User.update({ email: 'newemail@gmail.com' }, {
  where: { username: 'johndoe' }
});
console.log(updated);  // updated will be a boolean value indicating if the update was successful or not.
```

#### `destroy`

Used to destroy an instance in the database. It's useful when you want to delete an existing record from a table.

```javascript
const destroyed = await db.User.destroy({ where: { username: 'johndoe' } });
console.log(destroyed);  // destroyed will be a boolean value indicating if the deletion was successful or not.
```

*Please note* that all these examples are assuming that you have a `User` model defined in your `db` object. You would replace `User` with whatever model you're trying to interact with.


The `db` object also has two special properties: `sequelize` and `Sequelize`. `db.sequelize` is the instance of `Sequelize` that you're using to connect to the database, and `db.Sequelize` is the `Sequelize` library itself. see [[db.sequelize]] for more

In addition to these properties, the `db` object might also have other methods or properties, if you've added them to your models. For instance, if your `User` model has a method named `getFullName`, you could call it on an instance of the model like this: 
```js
const fullName = user.getFullName();
```


---

## The `index.js` file: 

The `models/index.js` file is the entry point for all your `Sequelize` models. It's designed to initialize and configure `Sequelize`, then import all your models and add them to a `db` object, which is then exported for use in your application. Let's go over this file line-by-line to understand what's happening.

```javascript
import fs from "fs";
import path from "path";
import Sequelize from "sequelize";
import process from "process";
import { fileURLToPath } from "url";
import { dirname } from "path";
```

First, we're importing several modules we'll need. [[fs and path]] are built-in `Node.js` modules for working with the **filesystem** and **file paths**, respectively. **`Sequelize`** is the `Sequelize` library itself, and [[process (Node.js)]] is a built-in `Node.js` module for accessing the system's environment variables.

```javascript
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
```

These lines are creating the `__filename` and `__dirname` variables, which are available by default in a **CommonJS** environment. Since we are using the ES6 `import/export` syntax, we need to create these variables manually. They represent the file path of the current module and the directory path of the current module, respectively.

```javascript
const env = process.env.NODE_ENV || "development";
const config = require(__dirname + "/../config/config.json")[env];
```

Here we're getting the current environment (`development`, `test`, or `production`) from the `NODE_ENV` environment variable, defaulting to `"development"` if it's not set. Then we're reading the database configuration for the current environment from `config/config.json`.

```javascript
const db = {};
```

Next, we create an empty `db` object. We'll be adding our models to this object as properties, and it will be exported at the end of the file.

```javascript
let sequelize;
if (config.use_env_variable) {
	sequelize = new Sequelize(process.env[config.use_env_variable], config);
} else {
	sequelize = new Sequelize(
		config.database,
		config.username,
		config.password,
		config
	);
}
```

We're initializing Sequelize here. If a `use_env_variable` property is set in our config file, we'll use it to get the database connection URI from the environment variables. If not, we'll use the `database`, `username`, `password`, and other properties from our config to initialize Sequelize.

```javascript
fs.readdirSync(__dirname)
	.filter((file) => {
		return (
			file.indexOf(".") !== 0 &&
			file !== basename &&
			file.slice(-3) === ".js" &&
			file.indexOf(".test.js") === -1
		);
	})
	.forEach((file) => {
		const model = require(path.join(__dirname, file)).default(
			sequelize,
			Sequelize.DataTypes
		);
		db[model.name] = model;
	});
```

Here we're reading all the files in the current directory, excluding index.js itself and any non-JavaScript files or test files. Each file should export a function that defines a Sequelize model. We import each model, pass the `sequelize` instance and `Sequelize.DataTypes` to it, and add the resulting model to the `db` object.

```javascript
Object.keys(db).forEach((modelName) => {
	if (db[modelName].associate) {
		db[modelName].associate(db);
	}
});
```

If our models have any associations defined, they should be in a method named `associate

` on the model. Here we're calling `associate` on each model and passing in the `db` object, which allows the models to reference each other to define their associations.

```javascript
db.sequelize = sequelize;
db.Sequelize = Sequelize;
```

Finally, we add `sequelize` (the instance) and `Sequelize` (the library) to the `db` object. These could be useful in other parts of your application (for instance, if you need to create a new instance of `Sequelize.QueryTypes`).

```javascript
export default db;
```

The `db` object is exported as the default export of this module.
