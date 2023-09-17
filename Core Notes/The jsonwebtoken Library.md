#seed 
###### upstream: 

## TLDR:

- `jsonwebtoken` provides an interface for creating, verifying and decoding JWTs.
- `jwt.sign` is used to create a new JWT.
- `jwt.verify` is used to verify a JWT and its signature.
- `jwt.decode` is used to decode a JWT without verifying its signature.
- Note: As of my knowledge cutoff in September 2021, please ensure to check the most recent `jsonwebtoken` library documentation for updates or changes.

### What is `jsonwebtoken`?

The `jsonwebtoken` library is a Node.js library that implements JSON Web Tokens (JWT). It provides a straightforward interface for creating, decoding, and verifying JWTs. Let's look at some of the main methods provided by this library:



### The `.sign()` method: 

*This method creates a new JWT. It takes the following parameters:*

- `payload`: This could be an object literal, buffer or string representing valid JSON.
- `secretOrPrivateKey`: This is a string, buffer, or object containing either the secret for HMAC algorithms or the PEM encoded private key for RSA and ECDSA.
- `options` (optional): An object literal containing the following available properties:
    - `algorithm` (default: `"HS256"`): Could be one of `"HS256"`, `"HS384"`, `"HS512"`, `"RS256"`, `"RS384"`, `"RS512"`, `"ES256"`, `"ES384"`, `"ES512"`, `"none"`.
    - `expiresIn`: Expressed in seconds or a string describing a time span [zeit/ms](https://www.npmjs.com/package/ms). Eg: `60`, `"2 days"`, `"10h"`, `"7d"`.
    - `notBefore`: Expressed in seconds or a string describing a time span [zeit/ms](https://www.npmjs.com/package/ms).
    - `audience`
    - `issuer`
    - `jwtid`
    - `subject`
    - `noTimestamp`
    - `header`
    - `keyid`
- `callback` (optional): If supplied, the callback is called with the `err` or the JWT.

### The `.verify()` method: 

*This method verifies a JWT. If the verification is successful, this method will return the payload decoded if the signature is valid. If not, it will throw an error.*

- `token`: This is the JWT string to verify.
- `secretOrPublicKey`: This is a string, buffer, or object containing either the secret for HMAC algorithms, or the PEM encoded public key for RSA and ECDSA.
- `options` (optional): An object literal containing the following available properties:
    - `algorithms`
    - `audience`
    - `issuer`
    - `ignoreExpiration`
    - `subject`
    - `clockTolerance`
    - `maxAge`
    - `clockTimestamp`
- `callback` (optional): If a callback is supplied, function will behave asynchronously, and call `callback` with the decoded payload or an error if the signature is invalid.

### The `.decode()` method: 

*This method decodes a JWT but does not verify it. It only decodes the payload without verifying whether the signature is valid.*

- `token`: This is the JWT string to decode.
- `options` (optional): An object literal containing the following available properties:
    - `json`: Force JSON.parse on the payload even if the header doesn't contain `"typ":"JWT"`.
    - `complete`: Return an object with the decoded payload and header.


  
### Example: 

see [[Storing Secret Keys in Environment Variables]]

```js
import express from 'express';
import jwt from 'jsonwebtoken';

const app = express();
const secretKey = 'mySecretKey'; // This should be stored securely

// Enable JSON parsing for POST requests
app.use(express.json());

// Dummy data, replace it with database in a real-world scenario
let users = [
  {
    username: 'john',
    password: 'password123'
  },
  // More users can be listed here...
];

// Simulating an authentication service
const authService = {
  authenticate: (username, password) => {
    const user = users.find(user => user.username === username && user.password === password);
    if (user) {
      return jwt.sign({ username: user.username }, secretKey, { expiresIn: '1h' });
    }
    return null;
  }
};

app.post('/login', (req, res) => {
  const { username, password } = req.body;
  const token = authService.authenticate(username, password);

  if (token) {
    // Send the token back to the client
    res.json({ token: token });
  } else {
    res.status(401).json({ message: 'Invalid credentials' });
  }
});

app.get('/protected', verifyToken, (req, res) => {
  // If it's valid, we can access the secured endpoint
  res.json({ message: 'You have accessed a protected route', user: req.user });
});

// This is a middleware function for checking the token
function verifyToken(req, res, next) {
  const bearerHeader = req.headers['authorization'];

  if (bearerHeader) {
    const bearer = bearerHeader.split(' ');
    const bearerToken = bearer[1];
    jwt.verify(bearerToken, secretKey, (err, data) => {
      if (err) {
        res.status(403).json({ message: 'Invalid or expired token' });
      } else {
        req.user = data; // The decoded payload (user object)
        next();
      }
    });
  } else {
    res.status(403).json({ message: 'No token provided' });
  }
}

app.listen(5000, () => console.log('Server started on port 5000'));
```
