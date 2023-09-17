#evergreen1 
###### upstream: 

## What's the difference between `.use()` and `.get()`? 

*...They both take in a request, parse it, and then respond with a response object right?*

Yes, both `.use()` and `.get()` methods in `Express.js` are used to define middleware functions. However, they are used in slightly different ways. 

### `app.use()`

More general case and is used to define a middleware function that will be executed for every request to the app. When you define a middleware function with `app.use()`, it can match all types of `HTTP` methods (`GET`, `POST`, etc.), and it will be executed for every incoming request, unless the function ends the request-response cycle or passes control to another middleware function.

It's often used for **middleware** that should apply **globally**, like logging, error handling, or body parsing. It can also be used to set up middleware for a subset of paths. *For example*
```js
app.use('/api', apiRouter)
```
would apply the middleware in `apiRouter` to all paths that start with **'/api'**.

### `app.get()` 
Used to define a middleware function that will **only** be executed for `HTTP GET` requests to a specific path. The `app.get()` method is part of a set of methods provided by `Express.js` for routing. Each of these methods corresponds to an HTTP method, and they're used to set up routes that respond to specific types of requests. 

### TLDR

In short, `app.use()` is more general, applying to all requests (or a subset of requests if a path is provided), while `app.get()` and its companion methods are more specific, only applying to requests of a specific type to a specific path.

## Does `.get()` require a path argument? 

*...And is the path argument optional for `.use()`?*

Yes, that's correct. The `app.get()` method in `Express.js` requires a path as its first argument. It defines a route that should respond to `GET` requests for that specific path. If you don't provide a path, `Express.js` won't know which requests should trigger the route.

On the other hand, `app.use()` is more flexible. You can call `app.use()` without a path, in which case the middleware function will apply to all incoming requests, regardless of the path or HTTP method. This is useful for setting up middleware that should run globally, like error handlers or body parsers.

But you can also call `app.use()` with a path as the first argument. When you do this, the middleware function will **only** apply to requests that match the specified path. This is useful when you want to apply specific middleware to a subset of your routes. For instance, you might want to apply an authentication middleware only to routes under '/api'.

**Here are some examples to illustrate this:**

```js
// Middleware applies to all requests
app.use((req, res, next) => {
  console.log('Time:', Date.now());
  next();
});

// Middleware applies only to requests with a path that starts with "/user"
app.use('/user', (req, res, next) => {
  console.log('Request URL:', req.originalUrl);
  next();
});

// Route applies only to GET requests to the "/user" path
app.get('/user', (req, res) => {
  res.send('GET request to the /user path');
});
```


## Does `get()` invoke the `next` argument like `use()` does? 

**No**, the `app.get()` method does not invoke the `next` argument. 

In Express, middleware functions are functions that have access to the `request` and `response` objects, as well as the `next` function. Middleware functions can perform tasks like modifying request or response objects, handling errors, or performing authentication.

By calling `next()` within a middleware function, you can continue the request-response cycle, passing control to subsequent middleware or route handlers.