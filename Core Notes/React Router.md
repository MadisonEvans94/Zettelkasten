#seed 
upstream:

---

**links**: 

---

Brain Dump: 

--- 




## Installation 

```sh
npm install react-router-dom
```

## Adding a Router

First thing to do is create a [Browser Router](https://reactrouter.com/en/main/routers/create-browser-router) and configure our first route. This will enable client side routing for our web app.

The `index.js` file is typically the entry point. Open it up and we'll put React Router on the page. Take a look at the import statements here: 

```jsx
import * as React from "react";

import * as ReactDOM from "react-dom/client";

import { BrowserRouter as Router } from "react-router-dom";

import "./index.css";

  

import App from "./App";

import reportWebVitals from "./reportWebVitals";

  

const root = ReactDOM.createRoot(document.getElementById("root"));

root.render(

<Router basename="/app">

<Routes>

<Route path="/" /> {/* 👈 Renders at /app/ */}

</Routes>

</Router>

);

  

// If you want to start measuring performance in your app, pass a function

// to log results (for example: reportWebVitals(console.log))

// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals

reportWebVitals();
```