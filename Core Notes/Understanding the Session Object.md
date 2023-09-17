#bookmark 
upstream:

---
**video links**: 
- [Your complete guide to understanding the express-session library](https://www.youtube.com/watch?v=J1qXK66k1y4&ab_channel=ZachGollwitzer)

---

`Express-session` is a middleware for `Express.js` that enables session functionality. The session object, which is a part of the request object, provides a way to persist data across requests.

In this document, we'll go over the basics of using express-session and delve into the details of the session object and its capabilities.

## Basic Usage

First, let's see how to use express-session in an Express app:

```javascript
import express from 'express';
import session from 'express-session';

const app = express();

app.use(session({
  secret: 'secret key',
  resave: false,
  saveUninitialized: true
}));
```

## The Session Object

Now, let's a deeper look at the session object:

```javascript
app.get('/', (req, res) => {
  if (req.session.views) {
    req.session.views++;
    res.send(`You visited this page ${req.session.views} times`);
  } else {
    req.session.views = 1;
    res.send('Welcome to this page for the first time!');
  }
});
```

In this route, we're using `req.session` to store and retrieve the number of times a user has viewed the page. On each request, we increment the number of views and send it back in the response.

## Session Object Properties and Methods

The session object comes with some built-in properties and methods:

### `secret`, `resave`, `saveUninitialized`
*These options are configurations for the `express-session` middleware.*

#### secret
This is a secret string that is used to sign the session ID cookie. This can either be a string for a single secret, or an array of multiple secrets. If an array of secrets is provided, only the first element will be used to sign the session ID cookie, while the entire array will be used to verify the signature in subsequent requests. This secret makes sure the session ID sent to the browser is secure and tamper-proof.

#### resave
This option forces the session to be saved back to the session store, even if the session was never modified during the request. Depending on your store this may be necessary, but it can also create race conditions where a client makes two parallel requests to your server and changes made to the session in one request may get overwritten when the other request ends, even if it made no changes (this behavior also depends on what store you're using).

#### saveUninitialized
This option forces a session that is "uninitialized" to be saved to the store. A session is considered "uninitialized" when it is new but not modified. Choosing `false` is useful for login sessions, reducing server storage usage, or complying with laws that require user permission before setting a cookie. Choosing `false` also helps with race conditions where a client makes multiple parallel requests without a session. 

*These options give you finer control over when and how sessions are saved, and should be chosen based on your specific use case.*

### **`req.session.id`** 
The **session ID** is read-only property that represents the session ID string. This string is stored in a web client's cookie.

### **`req.session.cookie`** 
This object contains the settings for the session ID cookie. You can use it to control the client-side behavior of the cookie.

### **`req.session.destroy(callback):`** 
This method deletes the session. It takes a callback function that will be called after the session is destroyed.

### **`req.session.reload(callback)`** 
This method reloads the session data from the store and re-populates the `req.session` object. It can be useful if you suspect the data has been changed outside the current request.

### **`req.session.save(callback)`** 
This method saves the session back into the session store. Even though the session middleware automatically saves the session, you can use this method if you want to force a save before you send the response.

### **`req.session.touch()`** 
This method resets the cookie maxAge to its original setting. It can be useful to keep the session alive.

## Storing Data

The session object can also store any arbitrary data:

```javascript
app.get('/', (req, res) => {
  req.session.user = {
    username: 'johndoe',
    password: 'secret'
  };

  res.send('User saved to session');
});
```

In this example, we're storing a user object in the session. This data will be available in `req.session.user` on subsequent requests.

## Log in Log out Example

Let's assume you have a basic Express application and you're using the `express-session` and `bcrypt` for password hashing. We'll create a basic **sign-in** system where a user can "sign in" with their username and password, creating a session, and then log out, which will destroy the session.

```javascript
import express from 'express';
import session from 'express-session';
import bcrypt from 'bcrypt';
import bodyParser from 'body-parser';

// Initialize express app
const app = express();

// use express-session middleware
app.use(session({
  secret: 'secret-key',
  resave: false,
  saveUninitialized: false
}));

// use body-parser middleware to handle form data
app.use(bodyParser.urlencoded({ extended: false }));

// For demonstration purposes, we'll store user data in an object
let users = {
  'johndoe': {
    password: '$2b$10$iuIUkj2JPtzIgANTv34aZeJRYsVncY5flVB/ywLFF8I2bdJReFX.m' // hashed password 'password'
  }
};

// Simple middleware to check if user is authenticated
function checkAuthenticated(req, res, next) {
  if (req.session.user) {
    next();
  } else {
    res.redirect('/login');
  }
}

app.get('/', checkAuthenticated, (req, res) => {
  res.send(`Hello, ${req.session.user}!`);
});

app.get('/login', (req, res) => {
  res.send(`
    <form method="post" action="/login">
      Username: <input type="text" name="username" required>
      Password: <input type="password" name="password" required>
      <button type="submit">Log in</button>
    </form>
  `);
});

app.post('/login', async (req, res) => {
  let username = req.body.username;
  let password = req.body.password;

  // Check if username exists and password is correct
  if (users[username] && await bcrypt.compare(password, users[username].password)) {
    // "Sign in" user by creating a session
    req.session.user = username;
    res.redirect('/');
  } else {
    res.send('Invalid username or password');
  }
});

app.get('/logout', (req, res) => {
  // "Sign out" user by destroying the session
  req.session.destroy(err => {
    if (err) {
      return res.redirect('/');
    }
    res.clearCookie('sid');
    res.redirect('/login');
  });
});

app.listen(3000, () => console.log('Server started on port 3000'));
```

In this code, we're creating a session when the user "signs in" and storing the username in `req.session.user`. We then use this session to keep the user signed in across requests. When the user "signs out", we destroy the session, effectively signing the user out. Note that we're using `body-parser` to handle form data.

The hashed password for 'johndoe' is a hash of the string 'password'. In a real application, you would not hard-code this data and would likely use a database to store user credentials. Also, you'd probably want to add more error checking and security features.

## Conclusion

In summary, express-session provides a way to store user data between HTTP requests. It uses a cookie to store a session ID and pairs this with a server-side store to keep track of data associated with that session ID. You can store almost any type of data and access it across requests, which is essential for many web application features like user authentication, form handling, and more.

Keep in mind that the session object, while convenient, should not be used to store large amounts of data or sensitive information. Always consider the implications for security and performance when deciding what to store in a session.
