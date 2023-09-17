#incubator 
###### upstream: [[Web Development]]

[Node.js Middleware](https://www.youtube.com/watch?v=_GJKAs7A0_4&ab_channel=TheNetNinja)

### What is it? 

[[Express]], and many other web frameworks like it, uses a series of **middleware functions** that process the incoming HTTP request. Each middleware function has access to the `req` (request), `res` (response), and `next` (a callback to call the next middleware in the chain) objects.


- A cool way to think about it is *"Middleware is like a conditioner for http requests that process the requests in user defined ways"*

![[Screen Shot 2023-06-19 at 12.29.22 PM.png]]
### Middleware Chains 

*I see. So requests come in and get filtered through a chain of middlewares so to speak. Is that an accurate statement? If so, what are some common middleware chains that might get used on different types of requests?*

Yes, you're correct! Incoming requests in Express.js get processed through a series of middleware functions. This middleware chain can be customized based on your application's needs, and the middleware functions are run in the order they are added to the application.

Some common middleware chains might look like the following:

1.  **Request logging and authentication (applied globally)**: Almost all applications will want to log incoming requests for debugging and analytics purposes. They might also want to check if a user is authenticated before processing the request further.
```js
app.use(morgan('dev')); // Log requests
app.use(myAuthenticationMiddleware); // Check if user is authenticated
```

2.  **Body parsing (applied globally)**: If your application is expecting JSON or URL-encoded data in the request body, you'll need middleware to parse that data and add it to the `req.body` object.
```js
app.use(express.json()); // Parse JSON bodies
app.use(express.urlencoded({ extended: true })); // Parse URL-encoded bodies
```

3.  **Route-specific middleware**: Sometimes, you might want to apply middleware only to certain routes. For example, you might want to validate the data in the request body for a POST request.
```js
app.post('/users', validateUserMiddleware, (req, res) => { 
  // Handle the request...
});
```

In this example, `validateUserMiddleware` might be a function that checks `req.body` to see if it has all the required properties to create a new user.

4.  **Error handling middleware (applied globally)**: At the end of your middleware chain, you might want to add a middleware function to handle any errors that were passed down through the chain.
```js
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).send('Something broke!');
});
```

### Is there a way to build my own middleware? If so, how would I do it?

Yes, you can definitely create your own middleware in Express.js. Middleware functions are simply functions that have access to the request (`req`), response (`res`), and next middleware (`next`) objects. They can execute any code, make changes to the request and response objects, end the request-response cycle, or call the next middleware function in the stack.

Here's an example of a very simple custom middleware function that logs the request method and URL to the console:

```javascript
const logRequestDetails = (req, res, next) => {
  console.log(`${req.method} Request to ${req.url}`);
  next(); // Important to call next() to continue to the next middleware
};

app.use(logRequestDetails);
```

In this example, the `logRequestDetails` function logs the request method (like `GET`, `POST`, etc.) and the URL of the request. After it's done, it calls `next()` to pass control to the next middleware function.

You can also define middleware functions that are specific to a certain route, like this:

```javascript
const logRequestDetails = (req, res, next) => {
  console.log(`Request to ${req.url}`);
  next();
};

app.get('/some_route', logRequestDetails, (req, res) => {
  res.send('This is some_route!');
});
```

In this case, `logRequestDetails` will only run for GET requests to `/some_route`.

You can make your middleware do all kinds of things – for example, it could check if a user is authenticated, validate and sanitize input, handle errors, serve static files, or anything else you can think of! The important thing is to remember to call `next()` when you're done, unless you're ending the request-response cycle by sending a response.
### Common Plug and Play Middleware for Different Frameworks 
- [[Express Middleware]]
- [[Ruby on Rails Middleware]]
- [[Django Middleware]]
- [[Flask Middleware]]

