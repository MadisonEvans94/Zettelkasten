#evergreen1 
###### upstream: [[Databases]]

### Definition

A **database migration** in the context of a web application involves managing changes and versions of database schemas. They are a way to apply version control to your database schema changes, just like how you use **Git** for version control of your code.

Migrations allow you to make changes to your database schema and apply these changes in a controlled manner on every server running the application.

The need for migrations arises from the fact that software requirements constantly change and, as a consequence, also the database design. 

---

### What is a Migration File?

A migration file contains the SQL statements for upgrading the database (e.g., `create table`, `add column`, etc) and for downgrading it (e.g., `drop table`, `drop column`). This allows you to move to newer versions of the schema and also revert to older ones.

---
### What Happens to Data Already in a DB When a Migration Is Ran?

The effect of a migration on existing data in a database can vary based on the type of migration operation being performed.

*Here are a few scenarios:*

#### 1. **Adding a column**: 

If a new column is added, existing records in the database will have a value of `null` (or a default value, if specified) for that column. Existing data is not otherwise affected.

#### 2. **Deleting a column**: 

If a column is deleted, all the data stored in that column for every record will be lost.

#### 3. **Changing a column type**: 

If the type of a column is changed, the impact on the data depends on the nature of the change. If the new type is compatible with the existing data (for instance, changing an integer to a float), the database will typically convert the existing data to the new type. If the new type is not compatible (for instance, changing a string to an integer), the existing data might be lost or result in an error.

#### 4. **Adding or removing a table**: 

If a new table is added, this has no impact on existing data in other tables. If a table is removed, all data within that table is deleted.

#### 5. **Adding constraints**: 

If you add a new constraint, like a unique constraint or a foreign key constraint, your existing data needs to comply with this new constraint. If the existing data violates the constraint, you'll likely get an error when you attempt to apply the migration.

#### 6. **Renaming a column or table**: 

If you rename a column or table, the data remains the same but the way you access it changes. The data in the column or table will be accessible under the new name.

*Remember...* migrations can be destructive (i.e., they can lead to data loss), especially when you're dropping tables or columns. Therefore, it's crucial to backup your data before running migrations, especially in a production environment. 

*Also...* thoroughly test your migrations in a development environment before running them in production.

### Migrations in `Sequelize`

[Database Migrations with Sequelize - Why and How](https://www.youtube.com/watch?v=4SySDaX753A&ab_channel=ChrisCourses)

`Sequelize` provides a utility to automatically create migration files and apply them. 

To get started, install the **`Sequelize CLI`**:

```bash
npm install --save-dev sequelize-cli
```

...then, initialize your project for `Sequelize`:

```bash
npx sequelize-cli init
```

This will create a few directories (including `migrations` and `models`), and a configuration file named `config.json` for database setup.

---

#### Creating a Migration

To create a migration, you can use the `sequelize-cli` command `migration:generate` and specify a name:

```bash
npx sequelize-cli migration:generate --name create-users
```

This will create a new migration file in the `migrations` directory. Open the new file and you will see the skeleton of a migration file:

```javascript
'use strict';
module.exports = {
  up: async (queryInterface, Sequelize) => {
    /* Code to run during migration goes here */
  },
  down: async (queryInterface, Sequelize) => {
    /* Code to undo the migration goes here */
  }
};
```

You can define your database changes in the `up` and `down` methods.

---

#### Running Migrations

After defining your migration, you can apply it using the `db:migrate` command:

```bash
npx sequelize-cli db:migrate
```

This will apply all pending migrations.

To undo the last migration, use the `db:migrate:undo` command:

```bash
npx sequelize-cli db:migrate:undo
```

---

#### Update vs Migrate

When we talk about databases, the term **"update"** often refers to updating records in the database. This can mean changing a user's email, adding a new post, etc. It's about manipulating the data within the tables.

On the other hand, **"migration"** is about structural changes to the database. This can involve creating or deleting tables, adding or removing columns, changing the type of a column, etc. It's about altering the schema of the database.

---

Database migrations are a complex topic and what I've written here is a basic introduction. For more detailed information and best practices, please refer to the [Sequelize Documentation](https://sequelize.org/master/manual/migrations.html) and the documentation of the database you are using.
