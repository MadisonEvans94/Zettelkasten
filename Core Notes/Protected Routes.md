#incubator 
upstream: [[React]]

---

**video links**: 

---

## Procedure

React Router is a popular library for routing in React applications. This tutorial will focus on creating protected routes in a React application using React Router v6.

### 1. Install Necessary Packages

First, you need to install React Router v6. You can do this by running the following command in your terminal:

```
npm install react-router-dom@6
```

### 2. Basic Routing

Here's a simple example of how you can set up routing in your application using React Router v6:

```jsx
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Home from "./Home";
import About from "./About";

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/about" element={<About />} />
      </Routes>
    </Router>
  );
}

export default App;
```

In the example above, we have two routes - a Home route and an About route. The path prop determines the URL path, and the element prop determines the component to render when the path matches the current URL.

### 3. Creating Protected Routes

Now let's say you want to add an authenticated route to your application. An authenticated route is a route that checks if a user is authenticated before rendering. If a user is not authenticated, they should be redirected to a different route (typically the login page).

To do this, you'll create a `PrivateRoute` component. This component will check if the user is authenticated. If they are, it renders the children prop. If not, it redirects the user to the login page.

```jsx
import { Navigate, useLocation } from 'react-router-dom';
import { useContext } from 'react';
import { UserContext } from '../Contexts/UserContext';

const PrivateRoute = ({ children }) => {
    const { isAuthenticated } = useContext(UserContext);
    const location = useLocation();

    return isAuthenticated
        ? children
        : <Navigate to="/login" state={{ from: location }} replace />;
};

export default PrivateRoute;
```

Now you can use the `PrivateRoute` component to protect your routes:

```jsx
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Home from "./Home";
import About from "./About";
import Dashboard from "./Dashboard";
import PrivateRoute from "./PrivateRoute";

function App() {
  const { isAuthenticated } = useContext(UserContext); 

  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/about" element={<About />} />
        <Route
          path="/dashboard"
          element={
            <PrivateRoute>
              <Dashboard />
            </PrivateRoute>
          }
        />
      </Routes>
    </Router>
  );
}

export default App;
```

### 4. Complete Example

For a complete example, let's create a simple application with public and private routes. For simplicity, let's just mock the authentication by using the useState hook:

```jsx
import React, { useState, useContext } from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { UserContext } from './Contexts/UserContext';
import PrivateRoute from './PrivateRoute';

// Mock pages
function Home() {
  return <h1>Home</h1>;
}

function About() {
  return <h1>About</h1>;
}

function Dashboard() {
  return <h1>Dashboard</h1>;
}

function Login() {
  return <h1>Login</h1>;
}

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);

  return (
    <UserContext.Provider value={{ isAuthenticated, setIsAuthenticated }}>
      <Router>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/about" element={<About />} />
          <Route path="/login" element={<Login />} />
          <Route
            path="/dashboard"
            element={
              <PrivateRoute>
                <Dashboard />
              </PrivateRoute>
            }
          />
        </Routes>
      </Router>
    </UserContext.Provider>
  );
}

export default App;
```

This is a basic example of how to implement protected routes in React using React Router v6. Depending on your application, you may need to add more advanced features, such as fetching user authentication status from an API, or saving user information in a global state using Redux or the Context API.
