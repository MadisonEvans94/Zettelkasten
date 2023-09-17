#incubator 
upstream:

---

**video links**: 

---

## Sequelize CLI: Understanding `migration:generate` and `db:migrate`

In `Sequelize`, migrations are a crucial part of database management, allowing you to define sets of changes to apply to your database schema.

There are two key commands related to migrations in `Sequelize CLI`: **`migration:generate`** and **`db:migrate`**.

### General Workflow 

Here is the typical `Sequelize` migration workflow:

#### 1. Define a model in your application.
This is done in the `./models` directory

#### 2. Run `npx sequelize-cli migration:generate --name create-some-entity` 
This creates a new migration file. This will create a blank migration file that you'll need to fill out.

#### 3. Fill out the `up` and `down` methods 
Fill out the `up` and `down` methods in the newly generated migration file. The `up` method should describe the changes to apply to the database, and the `down` method should describe how to undo these changes.

#### 4. Run `npx sequelize-cli db:migrate` 
This will apply the migration and update the database schema.

#### 5. If you make changes to the model
...you'll need to create a new migration file using `npx sequelize-cli migration:generate --name modify-some-entity` and fill out the `up` and `down` methods to describe these changes and their reversal.

#### 6. Run `npx sequelize-cli db:migrate` again 
...to apply the new migration

If you forget to create a new migration after changing a model **(step 5)** and run `npx sequelize-cli db:migrate` **(step 6)**, `Sequelize` won't throw an error, but your database schema will not be in sync with your models. This could lead to errors or unexpected behavior in your application when it tries to interact with the database using the updated models.

*see [[Sequelize Migrations - Filling out the Up and Down Methods]] for more info* 

### `migration:generate`

The `migration:generate` command creates a new migration file in your migrations directory.

#### What it does:

- Generates a new file with the necessary structure for writing a migration.
- Doesn't change your database.

#### When to use it: 

You should run `migration:generate` whenever you want to make a change to your database schema. For example, when:

- Creating a new table
- Modifying an existing table structure (e.g., adding, renaming, or removing columns)
- Removing an existing table

Remember, running `migration:generate` only creates a new migration file, it does not execute the migration. You will need to manually write the necessary code in the migration file to carry out the desired changes to the database schema.

Once you have created a new migration file (using `migration:generate`) and written the code to carry out the desired database changes, you can then run `db:migrate` to execute these changes.

#### How to use it:

You should specify a name for the migration when running this command:

```
npx sequelize-cli migration:generate --name create-users
```

The command above generates a new migration file called `YYYYMMDDHHmmss-create-users.js` (where `YYYYMMDDHHmmss` is replaced with the current date and time) in your migrations directory

### `db:migrate`

The `db:migrate` command executes the migrations you've defined.

*The command `npx sequelize-cli db:migrate` in Sequelize is equivalent to `flask db upgrade` in Flask. They both apply new migrations to the database*

#### What it does:

- Executes all migration files in your migrations directory in order of their timestamps.
- Applies any migrations that haven't been run yet.
- Modifies your database according to the instructions in the migrations.

#### When to use it: 

you should run `db:migrate`:

- After pulling changes from a version control system like Git where new migrations have been added.
- Whenever you want to update your database structure according to the latest migrations.
- Before the initial run of your application to ensure that the database structure is set up correctly.

In general, `db:migrate` is run more frequently than `migration:generate`, especially in development and testing environments. You would typically run `db:migrate` each time you change your database schema or receive new migrations from others working on the same project.

#### How to use it:

```
npx sequelize-cli db:migrate
```

The command above will apply all pending migrations to your database.

---

## Summary

In Sequelize, `migration:generate` is used to create new migration files, which define sets of changes to apply to your database schema. `db:migrate` is then used to apply those changes to your database.

---


