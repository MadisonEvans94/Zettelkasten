[Graph QL Full Series](https://www.youtube.com/watch?v=Y0lDGjwRYKw&list=PL4cUxeGkcC9iK6Qhn-QLcXCXPQUov1U7f&ab_channel=TheNetNinja)

#incubator 
upstream: [[Web API]]

---

**links**: 

---

Brain Dump: 
![[Traditional REST Paradigm.png]]
- downsides to REST 
	- under and over fetching 
	- when to use rest vs graphql 
--- 



Absolutely, let's create a succinct and organized markdown document covering the essential aspects of GraphQL for a developer. This document will provide clarity on what GraphQL is, why it's used, its tradeoffs compared to traditional API interfaces, and other relevant information.

---

## What is GraphQL?

GraphQL is a query language for APIs and a runtime for executing those queries by using a type system you define for your data. It was developed by Facebook in 2012 and later open-sourced. GraphQL isn't tied to any specific database or storage engine and is instead backed by your existing code and data.

### Key Concepts:

- **Queries**: Read operations in GraphQL, used to fetch the desired data structure.
- **Mutations**: Write operations, used for creating, updating, or deleting data.
- **Schema**: A model of the data that can be requested from the GraphQL server, including types, queries, and mutations.
- **Resolvers**: Functions that handle the logic for fetching the data for a specific query or mutation.

> see [[Queries and Mutations]] for more detail
### Analogy

#### Queries: 
Imagine you have a magic book that can answer any question about your favorite video game. You ask the book a question, like "What is the name of the main character?" or "How many levels are there?" The book answers with exactly what you want to know. In GraphQL, a **query** is like asking a question to a special computer program. You ask for specific information (like a list of your friends' names or the scores of recent games), and it gives you just that information, nothing more and nothing less.

#### Mutations: 
Now, think about when you want to change something in a game, like updating your high score, adding a new friend, or deleting an old game. In GraphQL, these changes are called **mutations**. It's like telling the magic book, "Hey, I have a new high score! Please update it." The book then changes the information to include your new high score.

#### Schema: 
A schema is like a map or a guide of all the things you can ask from the magic book. It tells you what questions are allowed and what kind of answers you can expect. For example, the schema might say you can ask about game characters, levels, and scores, and it will tell you exactly how you can ask these questions and what kind of answers you'll get. In GraphQL, the **schema** is a plan that shows all the types of queries and mutations you can do, and what kind of data, like numbers, text, or lists, you can ask for or change.

#### Resolvers: 
Resolvers are like helpers or guides in the magic book. When you ask a question (make a query) or want to change something (do a mutation), these guides find the answer or make the change for you. They know where to look and what to do to get you the information you need or update the information as you want. In GraphQL, **resolvers** are the functions in the system that know how to get the data you ask for in a query or how to make the changes you want in a mutation.

>So, in summary, using GraphQL is like having a magic book: you ask it questions (queries) or tell it to change things (mutations), it knows what you can ask and change (schema), and it has helpers to find answers and make changes for you (resolvers).

## Why Use GraphQL?

### Advantages:

1. **Data Aggregation from Multiple Sources**: GraphQL can aggregate data from multiple sources, including databases and third-party APIs.
2. **Fetch Only What You Need**: Unlike traditional REST APIs, GraphQL allows clients to specify exactly what data they need, reducing over-fetching.
3. **Strong Typing**: The schema defines the data structure, leading to more predictable and safer API interactions.
4. **Real-Time Data with Subscriptions**: GraphQL supports real-time data updates with subscriptions.
5. **Developer Experience**: Tools like GraphiQL provide a rich interface for exploring and testing GraphQL queries.

### Use Cases:

- Complex Systems with Many Entities and Relationships
- Projects Requiring Fine-grained Control Over Data Retrieval
- Applications that Need to Aggregate Data from Multiple Sources

## Tradeoffs Compared to Traditional APIs

### Compared to REST:

1. **Over-fetching and Under-fetching**: REST often leads to over-fetching (getting more data than needed) or under-fetching (making additional requests for related data). GraphQL solves this.
2. **Multiple Endpoints vs. Single Endpoint**: REST typically uses multiple endpoints for different data resources, whereas GraphQL uses a single endpoint.
3. **Caching**: REST APIs can leverage HTTP caching strategies more efficiently. Caching in GraphQL is more complex due to its dynamic nature.
4. **Error Handling**: REST uses HTTP status codes to indicate errors, while GraphQL uses a single status code, with errors included in the response body.
5. **Learning Curve**: GraphQL has a steeper learning curve compared to traditional REST APIs.

### Performance Considerations:

- **Query Complexity**: Complex GraphQL queries can be more demanding on the server.
- **Batching and Caching**: Strategies for batching requests and caching responses are different and can be more complex in GraphQL.

## Getting Started with GraphQL

### Creating a GraphQL API:

1. **Define a Schema**: Start by defining types, queries, and mutations in your schema.
2. **Implement Resolvers**: Create functions to fetch data for each field in the schema.
3. **Set Up a Server**: Use a GraphQL server library in your preferred programming language.

### Tools and Libraries:

- **GraphQL.js**: The reference implementation of GraphQL for JavaScript.
- **Apollo Server**: A popular GraphQL server library.
- **GraphiQL**: An in-browser IDE for exploring GraphQL.

## Best Practices

1. **Use Strongly Typed Schemas**: Ensure your schema accurately reflects the data model and business logic.
2. **Efficient Resolvers**: Optimize resolver functions to avoid performance bottlenecks.
3. **Error Handling**: Implement comprehensive error handling in your resolvers.
4. **Security**: Implement rate limiting, validation, and authentication/authorization mechanisms.
5. **Documentation**: Maintain clear and updated documentation for your GraphQL API.

## Conclusion

GraphQL offers a flexible, efficient approach to working with APIs. While it has many advantages, especially for complex applications, it also brings its own set of challenges and considerations. As with any technology choice, it's essential to evaluate its fit based on the specific needs and context of your project.

---

This document provides a foundational understanding of GraphQL, its advantages, tradeoffs, and some best practices for getting started. As a developer, it's crucial to dive deeper into each of these areas to fully leverage GraphQL's capabilities in your projects.