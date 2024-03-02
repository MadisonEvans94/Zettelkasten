#seed 
upstream: [[React]]

---

**links**: 

---
# Understanding Side Effects in React

In React, side effects (or "effects") are operations that can affect something outside the scope of the function being executed. These operations can include data fetching, subscriptions, or manually changing the DOM from React components. Unlike pure functions, which do not modify the external state or depend on it, side effects interact with the outside world.

## What are Side Effects?

Side effects are operations that:

- Affect external systems (e.g., making a fetch call to an API).
- Rely on values from external systems (e.g., subscribing to a WebSocket).
- Have observable interactions with the outside world (e.g., changing the DOM directly).

In a React component, operations like setting up a subscription or manually changing the DOM are side effects because they reach outside the component's own scope to interact with the outside world.

## Managing Side Effects in React

React provides a built-in hook, `useEffect`, to handle side effects in functional components. The `useEffect` hook lets you perform side effects in your components and gives you the capability to do them conditionally, based on what changes in your component's props or state.

### Basic Usage of `useEffect`

The `useEffect` hook takes two arguments:

1. **Effect Callback**: A function where you can place your side effect logic.
2. **Dependencies Array**: An optional array of dependencies that, if any change, the effect callback will rerun. If this array is empty (`[]`), the effect runs only once after the initial render.

#### Syntax

```jsx
useEffect(() => {
  // Side effect logic here.
}, [dependencies]);
```

### Example: Fetching Data

Fetching data from an API is a common side effect. Here's how you can do it with `useEffect`:

```jsx
import React, { useState, useEffect } from 'react';

function App() {
  const [data, setData] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      const response = await fetch('https://api.example.com/data');
      const result = await response.json();
      setData(result);
    };

    fetchData();
  }, []); // The empty array means this effect runs only once.

  return (
    <div>
      {data ? <div>{data}</div> : <div>Loading...</div>}
    </div>
  );
}
```

### Cleaning up Side Effects

Some effects require cleanup to avoid memory leaks (e.g., removing event listeners). You can return a cleanup function from the effect callback for this purpose:

```jsx
useEffect(() => {
  const subscription = dataSource.subscribe();
  return () => {
    // Clean up the subscription
    dataSource.unsubscribe(subscription);
  };
}, [dataSource]);
```

## Best Practices

- **Optimize Performance**: Use the dependencies array effectively to avoid unnecessary effect executions.
- **Cleanup**: Always clean up effects that set up subscriptions or listeners.
- **Isolation**: Keep effects focused on a single task for clarity and maintainability.

## Conclusion

Side effects are essential for interacting with the outside world in React applications. The `useEffect` hook provides a powerful and declarative way to manage these effects, helping you to write cleaner and more efficient code. Remember to use the dependencies array wisely and always clean up after your effects to prevent memory leaks.


