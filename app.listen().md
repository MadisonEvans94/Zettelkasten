#incubator 
###### upstream: 

### Description 

The `app.listen()` function in Express.js is used to bind and listen for connections on the specified host and port. This function is identical to Node's `http.Server.listen()` function.

### Syntax
```javascript
app.listen(port, [hostname], [backlog], [callback])
```

### Parameters
1. **port (Number/String):** This parameter is required and must be a number or a string. It specifies the port on which the server should run.

2. **hostname (String):** This is an optional parameter. It specifies the hostname of the server.

3. **backlog (Number):** This is also an optional parameter. It controls the maximum length of the pending connections queue. The actual length will be determined by your OS through its usual TCP/IP configuration. This is generally set between 511 and 65,535. Default value is 511 (not 512).

4. **callback (Function):** This is an optional parameter that specifies a function to be executed when the server starts listening.

### Usage
```javascript
const express = require('express');
const app = express();

app.listen(3000, function() {
  console.log('App listening on port 3000!');
});
```

In the above example, the Express app starts a server and listens on port 3000 for connections. The function passed as the second argument will execute once the server is ready to accept requests.

If you do not pass any argument to `app.listen()`, the server will start on the default HTTP port (80) or default HTTPS port (443) depending on whether the process has the necessary privileges.

### Notes
- The `app.listen()` function returns an `http.Server` object, making it possible to add additional listeners if needed.
- This method is a convenience method, because it's equivalent to creating an HTTP server separately and then using `app` as the callback function to handle requests. In fact, the following two code snippets do the same thing:

```javascript
const express = require('express');
const app = express();
const http = require('http');

http.createServer(app).listen(3000);
```

is equivalent to:

```javascript
const express = require('express');
const app = express();

app.listen(3000);
```

*To get further insight into the concept of "listening", see [[Understanding how Servers "listen"]]*