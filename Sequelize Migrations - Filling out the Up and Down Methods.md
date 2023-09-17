#seed 
upstream:

---

**video links**: 

---

The structure of a **migration** file contains two main functions: `up` and `down`.

## Creating a User Migration

Consider a `User` model as an example. Assume that we have a `User` model with `username` and `birthday` fields.

First, we generate a migration file via **`migration:generate`** command:

```bash
npx sequelize-cli migration:generate --name create-user
```

This will create a new file in the `migrations` directory with a name like `XXXXXXXXXXXXXX-create-user.js` (the Xs represent a timestamp, making each migration unique). The file contains boilerplate code for a migration:

```javascript
'use strict';
module.exports = {
  up: async (queryInterface, Sequelize) => {
    /* Code to create database table */
  },
  down: async (queryInterface, Sequelize) => {
    /* Code to drop database table */
  }
};
```

## The Up Method

The `up` method is used to make changes to the database schema, like creating a table, adding a column, or changing a data type. In the `up` method, we define what changes should be made to the database when the migration is run.

For our User model, let's define an `up` method to create a User table with `username` and `birthday` columns:

```javascript
up: async (queryInterface, Sequelize) => {
  return queryInterface.createTable('Users', {
    id: {
      allowNull: false,
      autoIncrement: true,
      primaryKey: true,
      type: Sequelize.INTEGER
    },
    username: {
      type: Sequelize.STRING,
      allowNull: false
    },
    birthday: {
      type: Sequelize.DATE
    },
    createdAt: {
      allowNull: false,
      type: Sequelize.DATE
    },
    updatedAt: {
      allowNull: false,
      type: Sequelize.DATE
    }
  });
}
```

## The Down Method

The `down` method should do the exact opposite of the `up` method. It describes how to revert the changes made in the `up` method. This might involve dropping a table, removing a column, etc.

For our User model, we can define a `down` method that drops the User table:

```javascript
down: async (queryInterface, Sequelize) => {
  return queryInterface.dropTable('Users');
}
```

So, the complete migration file will look like this:

```javascript
'use strict';
module.exports = {
  up: async (queryInterface, Sequelize) => {
    return queryInterface.createTable('Users', {
      id: {
        allowNull: false,
        autoIncrement: true,
        primaryKey: true,
        type: Sequelize.INTEGER
      },
      username: {
        type: Sequelize.STRING,
        allowNull: false
      },
      birthday: {
        type: Sequelize.DATE
      },
      createdAt: {
        allowNull: false,
        type: Sequelize.DATE
      },
      updatedAt: {
        allowNull: false,
        type: Sequelize.DATE
      }
    });
  },
  down: async (queryInterface, Sequelize) => {
    return queryInterface.dropTable('Users');
  }
};
```

This migration file creates a 'Users' table when the `up` method is run and drops the 'Users' table when the `down` method is run.

To execute this migration, we run `npx sequelize-cli db:migrate`, and to undo this migration, we use `npx sequelize-cli db:migrate:undo`.

## Making New Changes

If you want to add a new field to an existing table, you would use the `addColumn` method in the `up` function and `removeColumn` method in the `down` function. Here's an example of what your migration file might look like to add a "city" field to the Users table:

```javascript
'use strict';
module.exports = {
  up: async (queryInterface, Sequelize) => {
    return queryInterface.addColumn('Users', 'city', Sequelize.STRING);
  },

  down: async (queryInterface, Sequelize) => {
    return queryInterface.removeColumn('Users', 'city');
  }
};
```

In this file, the `up` function adds the 'city' column to the 'Users' table, and the `down` function removes it. This means that when you run this migration with `npx sequelize-cli db:migrate`, Sequelize will add the 'city' column, and when you revert the migration with `npx sequelize-cli db:migrate:undo`, Sequelize will remove the 'city' column.

This way, you can make changes to your database schema over time and easily revert those changes if necessary. Keep in mind that the `up` and `down` methods should always do the exact opposite of each other. In other words, whatever the `up` method does, the `down` method should undo.

see [[queryInterface]] for more 