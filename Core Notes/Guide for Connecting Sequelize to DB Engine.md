#incubator 

## Sequelize with MySQL / PostgreSQL

### 1. Install Dependencies 

To use `Sequelize` with **MySQL**:

```bash
npm install sequelize mysql2
```

Or with **PostgreSQL**:

```bash
npm install sequelize pg pg-hstore
```

### 2. Set Up Connection

You will need to set up a connection to your database. You will need to know the **name** of your database, and your **username** and **password**.

For **MySQL**:

```js
import { Sequelize } from 'sequelize';

const sequelize = new Sequelize('database', 'username', 'password', {
  host: 'localhost',
  dialect: 'mysql',
});
```

For **PostgreSQL**:

```js
import { Sequelize } from 'sequelize';

const sequelize = new Sequelize('database', 'username', 'password', {
  host: 'localhost',
  dialect: 'postgres',
});
```

### 3. Test Connection

You can use the `.authenticate()` method to test if the connection is successful:

```js
try {
  await sequelize.authenticate();
  console.log('Connection has been established successfully.');
} catch (error) {
  console.error('Unable to connect to the database:', error);
}
```

### 4. Define Models

You define models in the same way regardless of the dialect. *Here's an example:*

```js
import { DataTypes } from 'sequelize';

const User = sequelize.define('User', {
  username: DataTypes.STRING,
  birthday: DataTypes.DATE,
});
```

### 5. Sync Models with Database

After defining your models, you need to synchronize them with your database:

```js
await sequelize.sync();
```

This will create the necessary tables in your database if they do not exist. see [[sync vs migration]] for more

### 6. Persist and Query 

You can create and query data like this:

```js
const jane = await User.create({
  username: 'janedoe',
  birthday: new Date(1980, 6, 20),
});

const users = await User.findAll();
```

This is a very basic introduction. `Sequelize` is very feature-rich and allows for a lot of customization. For more details, see the [Sequelize Documentation](https://sequelize.org/master/).

---
Remember, if you're using a database that is not on your local machine (for example, on an EC2 instance or a database service), you will need to replace `'localhost'` with the address of your database server. Also, remember to never expose your database credentials publicly or in your codebase. **Always use environment variables** or some other form of secure configuration to handle sensitive data. See [[Understanding Environment Variables in AWS]] for more details. 