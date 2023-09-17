#incubator 
###### upstream: 

### Definition: 

A **Mutation** in GraphQL is a type of operation that can cause a change in the data on the server. Similar to how a query fetches data from the server, a mutation modifies data on the server and returns the newly modified data. In RESTful APIs, this would be similar to **POST**, **PUT**, **PATCH**, or **DELETE** methods.

### Analogy: 

*Now, let's break it down in simpler terms:*

Imagine you're playing a video game where you have a character, and this character has some items in their backpack. Now, you decide to pick up a new item - say, a health potion. When you do this, you're changing (or "mutating") the contents of your character's backpack.

In this analogy:

-   The video game server is like the GraphQL server.
-   The character's backpack is like the database on the server.
-   The action of picking up the health potion is like a GraphQL mutation.

So when you tell the game to add the health potion to your backpack (or when you tell the GraphQL server to make a mutation), two things happen:

1.  The game changes the contents of the backpack to include the health potion (or the server changes the database).
2.  The game shows you the updated contents of the backpack, including the new health potion (or the server sends back the updated data).

Here's what a mutation might look like in GraphQL:
```graphql
mutation {
  addItemToBackpack(item: "Health Potion") {
    updatedBackpack
  }
}
```

In this mutation, you're telling the server to run the `addItemToBackpack` operation, with `"Health Potion"` as the item to add. After this operation runs, it returns the `updatedBackpack`, which will include the new item.

### Additional Questions: 

*is the `addItemToBackpack` a function that I the programmer would have to define on the backend? If so, what might that function look like?*

Yes, `addItemToBackpack` refers to a **resolver** function that you would define on the server side (see [[Writing Resolvers]] for more details). This function is responsible for how the mutation operation `addItemToBackpack` is carried out.

Resolvers in GraphQL are functions that handle the data fetching for your types and fields. For mutations, the resolver function also needs to handle how the data is changed or "mutated" in the data source (database, in this case). So to compare to the RESTful workflow, a resolver is like an ORM function that is responsible for updating db data. 

Here's an example of what the `addItemToBackpack` resolver function might look like, using a Node.js environment, a hypothetical database model `Backpack`, and a MongoDB-like database interface:

```javascript
addItemToBackpack: async (_, { item }, context) => {
  // In this example, we're assuming that we've already authenticated
  // the user and attached their data to the `context`.
  const userId = context.user.id;

  // Fetch the user's backpack from the database.
  const backpack = await Backpack.findOne({ userId });

  // Add the item to the backpack.
  backpack.items.push(item);

  // Save the updated backpack to the database.
  const updatedBackpack = await backpack.save();

  // Return the updated backpack.
  return updatedBackpack;
}
```

In this resolver:

-   The first argument (`_`) is a placeholder for parent data, which isn't used in this root-level resolver.
-   The second argument (`{ item }`) contains the input data from the mutation, which in this case is the item to add to the backpack.
-   The third argument (`context`) contains contextual information like the currently logged-in user.

This is a fairly simplified example, and a real resolver function might need to handle things like validating input data, error handling, and possibly complex data transformations. But it should give you a basic idea of how a resolver function for a mutation might be implemented.