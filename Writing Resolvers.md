#incubator 
###### upstream: [[GraphQL]]

### Origin of Thought:

**What is GraphQL?** GraphQL is a query language designed to build client applications by providing an intuitive and efficient way to fetch data. Unlike traditional REST APIs, with GraphQL, you can request exactly what you need and nothing more. This minimizes the amount of data transferred over the network and allows the client to control the shape of the response.

### Basic Concepts: 
1.  **Schema**: This is a model of the data that can be fetched through the GraphQL server. It defines the types of data, their relationships, and how they can be retrieved.
    
2.  **Types**: GraphQL schema is built with types. There are various built-in types like `String`, `Int`, `Float`, `Boolean`, and `ID`, as well as custom types.
    
3.  **Query**: A query is a read-only fetch operation in GraphQL. In a query, you specify the fields you want to retrieve.
    
4.  **Mutation**: A mutation is a write followed by a fetch. It modifies data in the data source and returns the new value.
    
5.  **Subscription**: Subscriptions allow the server to send data to its clients when a specific event happens.
    
6.  **Resolver**: Resolvers provide the instructions for turning a GraphQL operation into data. They resolve the query to the actual data.

### Syntax: 

A basic query might look something like this:
```graphql
query {
  user {
    name
    email
  }
}
```

This query requests for a `user` and specifies that it wants the `name` and `email` of the user. The server responds with a JSON object where these properties are filled with the corresponding values:
```json
{
  "data": {
    "user": {
      "name": "John Doe",
      "email": "john.doe@example.com"
    }
  }
}
```

**GraphQL Mutation Syntax**

A mutation in GraphQL might look like this:
```graphql
mutation {
  addUser(name: "John", email: "john@example.com") {
    id
    name
    email
  }
}
```

This mutation is requesting to add a user, and then fetch the `id`, `name`, and `email` of the new user.

**GraphQL Subscription Syntax**

A subscription in GraphQL might look like this:

```graphql
subscription {
  userAdded {
    id
    name
    email
  }
}
```

This subscription is listening for when a new user is added. When a new user is added, it sends the `id`, `name`, and `email` of the new user to the client.

This should give you a start in understanding the syntax of GraphQL. The best way to learn more would be to experiment with creating your own GraphQL server, defining your own types and resolvers, and then making queries, mutations, and subscriptions against your server. There are also a lot of online playgrounds where you can experiment with GraphQL without setting up a server.

### Additional Questions: 

*Let's say I have a database that is set up to work with a REST api, but I want to transition to graphql. Do I have to change anything with the database schema or am I able to decouple this process and just update the api server?*

Transitioning from a REST API to a GraphQL API usually doesn't require changes to your database schema. The primary job of both REST and GraphQL is to expose your database to your client applications, and they can work with a variety of database schemas.

The transition primarily involves changing the way you expose your data to your clients. Instead of multiple endpoints that return fixed data structures (as in REST), you will expose a single endpoint and define a GraphQL schema that models your data.

Resolvers are functions that you write to populate the fields in your schema. They can retrieve data from anywhere, including a SQL or NoSQL database, a microservice, a third-party API, or even a file system. The resolver functions translate between the schema fields and your database.

To put it in practical terms, here's a simplified step-by-step process:

1.  **Define Your GraphQL Schema**: Define types in your schema that correspond to the entities in your database. Each of these types should include the fields that clients might need to access.
    
2.  **Write Resolvers**: For each type and field in your schema, define a resolver function. These functions specify how to fetch the underlying data. For example, to retrieve data from a SQL database, a resolver might contain SQL queries.
    
3.  **Create the GraphQL Server**: Set up your GraphQL server using a library such as Apollo Server or express-graphql. This server is where you tie together your schema and resolvers.
    
4.  **Test Your Server**: Make sure that your server works correctly by sending it some test queries and mutations.
    
5.  **Replace REST API Calls**: In your client application, replace REST API calls with equivalent GraphQL queries or mutations. Since GraphQL can fetch all necessary data in one request, you might be able to replace multiple REST calls with a single GraphQL call.
    

Remember, [[migrating from REST to GraphQL]] can be a complex task (see the link for more details), especially for larger APIs, so it's often a good idea to transition gradually. You might start by exposing some of your data through a GraphQL API, while leaving the rest of your data to be accessed through your existing REST API. As you get more comfortable with GraphQL, you can move more of your data fetching over to it.


