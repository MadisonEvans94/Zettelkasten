#incubator 
upstream:

## Sync vs Migration

**Synchronization** and **migration** serve similar purposes – *to propagate the changes in your models to the actual database* – but they are used in different scenarios and have different capabilities:

### `sync()`: 

...is straightforward and automatic

it makes the database match your `Sequelize` models by **creating** or **dropping** tables. 

It's useful in development or for small projects where you have full control over the database and it's okay to lose existing data.

### Migrations

...on the other hand, provide more control and precision. 

They are a set of sequentially ordered scripts that describe the changes to your tables over time. Each migration file describes the changes needed to go from one version of your database to the next. This means you can update your database incrementally, without losing data, and you can also rollback changes if something goes wrong.