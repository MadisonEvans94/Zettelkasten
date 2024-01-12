#evergreen1 
upstream: [[React]], [[GraphQL]]

---

**links**:
- [official docs](https://www.apollographql.com/docs/react/)
- [video guide](https://www.youtube.com/watch?v=lRKWJtzqwcQ&pp=ygUNYXBvbGxvIHJlYWN0IA%3D%3D)


---

Brain Dump: 

--- 


## What Is Apollo? 

**Apollo Client** is a comprehensive state management library for JavaScript that enables you to manage both local and remote data with GraphQL. Use it to fetch, cache, and modify application data, all while automatically updating your UI.

Apollo Client helps you structure code in an economical, predictable, and declarative way that's consistent with modern development practices. The core `@apollo/client` library provides built-in integration with React, and the larger Apollo community maintains [integrations for other popular view layers](https://www.apollographql.com/docs/react/#community-integrations).

## Features 

- **Declarative data fetching:** Write a query and receive data without manually tracking loading states.
- **Excellent developer experience:** Enjoy helpful tooling for TypeScript, Chrome / Firefox devtools, and VS Code.
- **Designed for modern React:** Take advantage of the latest React features, such as hooks.
- **Incrementally adoptable:** Drop Apollo into any JavaScript app and incorporate it feature by feature.
- **Universally compatible:** Use any build setup and any GraphQL API.
- **Community driven:** Share knowledge with thousands of developers in the GraphQL community.

## Quick Guide 

### Installation 

Applications that use Apollo Client require two top-level dependencies:

- `@apollo/client`: This single package contains virtually everything you need to set up Apollo Client. It includes the in-memory cache, local state management, error handling, and a React-based view layer.
- `graphql`: This package provides logic for parsing GraphQL queries.

Run the following command to install both of these packages:

```bash
npm install @apollo/client graphql
```

### Imports

With our dependencies set up, we can now initialize an `ApolloClient` instance.

In `index.js`, let's first import the symbols we need from `@apollo/client`:

```js
import { ApolloClient, InMemoryCache, ApolloProvider, gql } from '@apollo/client';
```

Next we'll initialize `ApolloClient`, passing its constructor a configuration object with the `uri` and `cache` fields:

```js
const client = new ApolloClient({ 
	uri: 'https://flyby-router-demo.herokuapp.com/', 
	cache: new InMemoryCache(), 
});
```

- `uri` specifies the URL of our GraphQL server.
- `cache` is an instance of [`InMemoryCache`](https://www.apollographql.com/docs/react/api/cache/InMemoryCache/), which Apollo Client uses to cache query results after fetching them. 

> see [Caching in Apollo](https://www.apollographql.com/docs/react/caching/overview) for more details 

That's it! Our `client` is ready to start fetching data.
### Provider 

You connect Apollo Client to React with the `ApolloProvider` component. Similar to [[React Context Provider]] [`Context.Provider`](https://react.dev/reference/react/useContext), `ApolloProvider` wraps your React app and places Apollo Client on the context, enabling you to access it from anywhere in your component tree.

In `index.js`, let's wrap our React app with an `ApolloProvider`. We suggest putting the `ApolloProvider` somewhere high in your app, above any component that might need to access GraphQL data.

```jsx
import React from 'react'; 
import * as ReactDOM from 'react-dom/client'; 
import { ApolloClient, InMemoryCache, ApolloProvider } from '@apollo/client'; 
import App from './App'; 

const client = new ApolloClient({ 
	uri: 'https://flyby-router-demo.herokuapp.com/', 
	cache: new InMemoryCache(), 
}); 

// Supported in React 18+ 
const root = ReactDOM.createRoot(document.getElementById('root')); 

root.render( 
	<ApolloProvider client={client}> 
		<App /> 
	</ApolloProvider>, 
);

```

### Client Object 

#### What is the Apollo Client Object?

The Apollo Client object is a central piece of the Apollo library. It's the core that manages both the interaction with your GraphQL API and the in-memory cache of your application. This object is responsible for sending queries, handling responses, and managing local state.

#### Why Does it Exist?

Apollo Client is designed to simplify the process of interacting with a GraphQL API from your application. It abstracts away the complexities of network requests, caching, and UI updates, providing a more streamlined and efficient way to handle data in modern web applications.

#### Important Functions

- **Querying and Mutations:** `ApolloClient` provides methods like `query()` and `mutate()` to fetch and update data respectively.
- **Cache Management:** With its `InMemoryCache`, Apollo Client caches query results to improve performance and enable offline support.
- **Error Handling:** It offers robust error handling capabilities to gracefully manage API and network errors.
- **Subscriptions:** For real-time data, Apollo Client integrates GraphQL subscriptions to listen to data updates.

### Fetching Data With `useQuery` 

After your `ApolloProvider` is hooked up, you can start requesting data with `useQuery`. The `useQuery` hook is a [React hook](https://react.dev/reference/react) that shares GraphQL data with your UI.

Switching over to our `App.js` file, we'll start by replacing our existing file contents with the code snippet below:

```jsx
// Import everything needed to use the `useQuery` hook 
import { useQuery, gql } from '@apollo/client'; 

export default function App() { 
	return ( 
		<div> 
			<h2>My first Apollo app 🚀</h2> 
		</div> 
	); 
}
```

We can define the query we want to execute by wrapping it in the `gql` template literal:

```js
const GET_LOCATIONS = gql` 
	query GetLocations { 
		locations { 
			id 
			name 
			description 
			photo 
		} 
	} 
`;
```

Next, let's define a component named `DisplayLocations` that executes our `GET_LOCATIONS` query with the `useQuery` hook:

```jsx
function DisplayLocations() { 
	const { loading, error, data } = useQuery(GET_LOCATIONS); 
	
	if (loading) return <p>Loading...</p>; 
	if (error) return <p>Error : {error.message}</p>; 
	
	return data.locations.map(({ id, name, description, photo }) => ( 
		<div key={id}> 
			<h3>{name}</h3> 
			<img width="400" height="250" alt="location-reference" src={`${photo}`} /> 
			<br /> 
			<b>About this location:</b> 
			<p>{description}</p> 
			<br /> 
		</div> 
	)); 
}
```

Whenever this component renders, the `useQuery` hook automatically executes our query and returns a result object containing `loading`, `error`, and `data` properties:

- Apollo Client automatically tracks a query's loading and error states, which are reflected in the `loading` and `error` properties.
- When the result of your query comes back, it's attached to the `data` property.

Finally, we'll add `DisplayLocations` to our existing component tree:

```jsx
export default function App() { 
	return ( 
		<div> 
			<h2>My first Apollo app 🚀</h2> 
			<br/> 
			<DisplayLocations /> 
		</div> 
	);
}
```

When your app reloads, you should briefly see a loading indicator, followed by a list of locations and details about those locations! If you don't, you can compare your code against the [completed app on CodeSandbox](https://codesandbox.io/s/github/apollographql/docs-examples/tree/main/apollo-client/v3/getting-started).

Congrats, you just made your first component that renders with GraphQL data from Apollo Client! 🎉 Now you can try building more components that use `useQuery` and experiment with the concepts you just learned.
### Folder Structure 

#### Best Practices for Organizing Queries and Mutations

To maintain a clean and modular codebase, it is best practice to separate queries and mutations into different directories. This separation enhances readability and maintainability of your code. 

#### Structure Overview

- **Queries Directory:** Store all your GraphQL query definitions here. You can further organize them into subdirectories based on features or data types.
- **Mutations Directory:** Similarly, keep all mutation definitions in this directory. Organizing them based on the data they modify or the feature they are part of is a good practice.
- **Subscriptions (optional):** If using subscriptions, having a dedicated directory for them follows the same logic of separation for clarity and maintainability.
- **Utility Functions:** It's also beneficial to have a directory for utility functions related to GraphQL operations, like error handling or data transformation functions.

#### Example Structure

```
src/
|-- graphql/
    |-- queries/
        |-- userQueries.js
        |-- productQueries.js
    |-- mutations/
        |-- userMutations.js
        |-- productMutations.js
    |-- subscriptions/
        |-- userSubscriptions.js
    |-- utils/
        |-- graphqlErrorHandler.js
```

This structure ensures that your GraphQL operations are organized and easily accessible, making your codebase easier to navigate and work with.




