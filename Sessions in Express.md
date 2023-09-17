#seed 

upstream: [[Web Sessions - An Extensive Guide]]

## Introduction

**Session Management** is a critical aspect of web development. It allows you to store user data between HTTP requests. Unlike cookies, session data is stored on the server-side, making it more secure.

**express-session** is a popular session middleware for `Express.js`. It creates a session middleware with the given options.

## Workflow
### Installation 

You can install `express-session` using `npm`:

```bash
npm install express-session
```

Import and use the `express-session` middleware as follows:

```javascript
var express = require('express');
var session = require('express-session');

var app = express();

app.use(session({
  secret: 'your_secret_value_here',
  resave: false,
  saveUninitialized: true
}));

//... Rest of the Express app configuration
```

*see [[Understanding the Session Object]] for more info*

### Configuring the session

You can access the session data in routes via `req.session`. Here's an example:

```javascript
app.get('/', function(req, res, next) {
  if(req.session.views) {
    req.session.views++;
    res.send("You visited this page " + req.session.views + " times");
  } else {
    req.session.views = 1;
    res.send("Welcome to this page for the first time!");
  }
});
```

In this example, a `views` count is stored in the session. When the user revisits the page, the count is incremented.

### Save and retrieve sessions

You can add data to the session simply by adding properties to the `req.session` object as shown in the previous example. To retrieve the data, you just need to access the property from `req.session`.

```javascript
// Save data to session
app.get('/save', function(req, res, next) {
  req.session.data = "This is some saved data";
  res.send("Data saved to session");
});

// Retrieve data from session
app.get('/retrieve', function(req, res, next) {
  var data = req.session.data;
  res.send("Retrieved data from session: " + data);
});
```

### Destroy a session

You can destroy the session by calling `req.session.destroy()` method:

```javascript
app.get('/logout', function(req, res, next) {
  req.session.destroy(function(err) {
    if(err) {
      return console.log(err);
    }
    res.send("Session destroyed");
  });
});
```

### Session Options

The `session()` function takes an options object. Here are some of the options:

- `secret`: This is a secret key used for signing the session ID cookie. It is required.
- `name`: The name of the session ID cookie. Default is 'connect.sid'.
- `resave`: This option forces the session to be saved back to the session store, even if the session was never modified during the request. Default is `false`.
- `saveUninitialized`: This forces a session that is "uninitialized" to be saved to the store. A session is uninitialized when it is new but not modified. Default is `true`.
- `cookie`: This sets the session cookie settings. It can have options like `maxAge`, `secure`, etc.
	- `maxAge`: The maximum age (in milliseconds) of a valid session.
	- `secure`: If `true`, the cookie will only be sent over HTTPS.
	- `httpOnly`: If `true`, the cookie is only accessible over HTTP(S), not client JavaScript, helping to prevent cross-site scripting attacks.

Here's an example of using these options:

```javascript
app.use(session({
  secret: 'your_secret_key',
  name: 'cookie_name',
  resave: false,
  saveUninitialized: true,
  cookie: { 
    secure: true,
    maxAge: 60000,
    httpOnly: true
  }
}));
```

### Using a session store

By default, **`express-session`** stores the session data in memory, which is **not** ideal for a production environment. In production, you should use a session store such as `connect-redis`, `connect-mongodb-session`, `connect-mongo`, etc. For this example we'll use **`connect-redis`** *(see [[Redis]] for more background if needed)*

*First, install `connect-redis` and `redis`:*

```bash
npm install connect-redis redis
```

*Then, use it as your session store:*

```javascript
var express = require('express');
var session = require('express-session');
var RedisStore = require('connect-redis')(session);

var app = express();

app.use(session({
  store: new RedisStore({
    host: 'localhost', 
    port: 6379, 
    client: redisClient, 
    ttl: 86400
  }),
  secret: 'your_secret_key',
  resave: false,
  saveUninitialized: false,
  cookie: { secure: true }
}));

// ... rest of the Express app configuration
```

*Remember*, using a session store requires an additional database which you have to manage, backup, etc. It may also increase your application's complexity and cost. You should carefully consider these factors when deciding whether to use a session store.