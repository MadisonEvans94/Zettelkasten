#seed 
###### upstream: 

### Definition: 

**What is Routing?**

Routing in the context of web development refers to the mechanism by which a web application decides how to respond to a client request to a particular endpoint, which is a URI (or path), and a specific HTTP request method (GET, POST, PUT, DELETE, etc.). Each route can have one or more handler functions, which are executed when the route is matched.

### Example - Key Points About Routing in Express.js:

1.  **Basic Routing**: Basic routing in Express.js involves defining routes for specific HTTP methods. For example, a basic GET request would look like this:
```js
app.get('/', function (req, res) {
  res.send('Hello World')
})
```

In this example, the application responds with 'Hello World' for requests to the root URL ('/') or route.

2.  **Route Path**: Besides the root URL, routes can also have path parameters. For example, in `app.get('/users/:userId', callback)`, `:userId` is a path parameter and can be accessed by `req.params.userId` in the callback function.
    
3.  **Route Handlers**: You can provide multiple callback functions that behave like middleware to handle a request. The only exception is that these callbacks might invoke `next('route')` to bypass the remaining route callbacks. This can be useful to conditionally run more than one route handler for a particular path, depending on the conditions of the request.

```js
app.get('/example/b', function (req, res, next) {
  console.log('the response will be sent by the next function ...')
  next()
}, function (req, res) {
  res.send('Hello from B!')
})
```

4.  **Route Methods**: Express.js uses HTTP method names as a shorthand for routing. For example, app.get() handles GET requests and app.post() handles POST requests. To handle all HTTP methods at a specific route, you can use app.all().
    
5.  **Route Files and Modularization**: For complex applications, routes can be broken down into multiple files and imported into the main application. The [[express.Router()]] function can be used to create modular mountable route handlers.
    
6.  **Wildcard Routes**: Wildcard characters can be used in routes. These can be helpful for sending custom 404 pages or handling multiple similar routes.
    
7.  **Order of Routes**: The order of routes matters. The first matching route will handle the request, so more specific routes should be placed before more general ones.
    
8.  **Middleware in Routes**: Middleware functions can be applied to routes. Middleware functions can perform tasks like error handling, logging, validating and sanitizing input, or user authentication.\

### The `req` and `res` arguments: 

*In the line `function (req, res)`, what are the `req` and `res` arguments and where do they come from?*

In most web frameworks, when a client (*such as a web browser or another server*) sends an HTTP request, the request comes loaded with information which is automatically encapsulated in the `req` (request) object by the web framework. This request object is then passed as an argument to the relevant callback function. Simultaneously, Express generates a `res` (response) object that represents what will be sent back to the client. This response object is also passed into the callback function and can be modified within the function to shape the outgoing response.

*Here's a simplified representation of how things might look under the hood:*

```js
function get(path, callback) {
    // When a GET request is made to the path...
    const req = {
        //... populate this object with request data
    };

    const res = {
        // ... provide methods for crafting a response
    };

    callback(req, res);
}
```

In actual execution, `req` and `res` are not simply empty objects. They are pre-populated by the `Express.js` framework (and many other frameworks) with many properties and methods corresponding to the incoming HTTP request and the outgoing HTTP response, respectively.

For example, the `req` object has properties like `req.body`, `req.params`, `req.query`, and more that are filled in by Express.js depending on the details of the HTTP request (like the URL, headers, query parameters, request body, etc.). The `req` object also has methods that allow you to access cookies, check authentication, and more.

Similarly, the `res` object comes with numerous methods that allow you to control what's sent back in the HTTP response. For example, `res.send()`, `res.json()`, `res.render()`, `res.redirect()`, and more.

When a request comes in, Express.js will create `req` and `res` objects, populate them appropriately for that request, and then pass them as arguments to your route handler function. So while the objects aren't empty, you as a developer don't have to manually fill these objects — Express.js does that for you.

The simplified representation given in the previous response was meant to convey the general flow and not the specifics of what `req` and `res` are.

#### How Express "fills" in the parameters of the `req` and `res` arguments... 

Express.js uses a series of middleware functions that process the incoming HTTP request. Each middleware function has access to the `req` (request), `res` (response), and `next` (a callback to call the next middleware in the chain) objects.

When a request comes in, Express.js creates the `req` and `res` objects and then passes them through each middleware function in the order they were registered (added to the application). Each middleware function can read from and modify the `req` and `res` objects.

For example, a very common middleware is the `body-parser` middleware. This middleware looks at the incoming HTTP request, and if it has a body, the middleware will read the body data (like form data or JSON data), parse it, and then add it to the `req.body` property.

Here's a simplified version of what the `body-parser` middleware might look like:

```js
function bodyParser(req, res, next) {
  let body = '';
  
  req.on('data', chunk => {
    body += chunk.toString(); // append each chunk of data to the body string
  });

  req.on('end', () => {
    req.body = JSON.parse(body); // when the request is done, parse the body and add it to the req object
    next(); // call the next middleware
  });
}
```

