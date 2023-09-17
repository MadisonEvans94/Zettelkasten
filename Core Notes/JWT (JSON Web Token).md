#seed 
###### upstream: 

JSON Web Tokens (JWT) is an open standard (RFC 7519) that defines a compact and self-contained way for securely transmitting information between parties as a JSON object. This information can be verified and trusted because it is digitally signed.

### How Does JWT Work?

A JWT is composed of three parts: a header, a payload, and a signature. These are separated by dots (.), and they form a string that looks like this:

```plaintext
xxxxx.yyyyy.zzzzz
```

1. **Header**: The header typically consists of two parts: the type of the token, which is JWT, and the signing algorithm being used, such as HMAC SHA256 or RSA.

```json
{
  "alg": "HS256",
  "typ": "JWT"
}
```

2. **Payload**: The second part of the token is the payload, which contains the "claims". Claims are statements about an entity (typically, the user) and additional data. There are three types of claims: registered, public, and private claims.

```json
{
  "sub": "1234567890",
  "name": "John Doe",
  "admin": true
}
```

3. **Signature**: To create the signature part you have to take the encoded header, the encoded payload, a secret, the algorithm specified in the header, and sign that. The signature is used to verify the message wasn't changed along the way, and, in the case of tokens signed with a private key, it can also verify that the sender of the JWT is who it says it is.

```javascript
HMACSHA256(
  base64UrlEncode(header) + "." +
  base64UrlEncode(payload),
  secret)
```

### Why use JWT?

JWTs can be signed using a secret (with the HMAC algorithm) or a public/private key pair using RSA or ECDSA. They are designed to be compact, URL-safe, and usable in a web context. 

Some advantages of JWT include:

- **Self-contained**: The payload contains all the required information about the user, avoiding the need to query the database more than once.
- **Compact**: Because of their smaller size, JWTs can be sent through URLs, POST parameters, or inside HTTP headers. This makes JWT a good choice for authentication through single sign-on.
- **Usage across different domains**: JWTs work across different domains, so they are useful in federated identity scenarios.

### Trade-offs of using JWT:

- **Token size**: Because JWTs can include additional claims, the size of the JWT can become much larger than traditional session IDs.
- **Stateless (pro and con)**: JWTs are stateless, meaning that you should include all necessary information in the payload and no session state is kept on the server. This can be a benefit because it reduces the load on your server, but it can also be a disadvantage as you won't be able to invalidate a session without extra logic on your server.
- **Expiration**: You must handle the token expiration scenario. Once a JWT is issued, it's valid until it expires, there's no way of invalidating it. So if it gets stolen, it can be used until it expires.
- **Storage on the client side**: JWTs are usually stored in the browser's LocalStorage, and are vulnerable to XSS attacks. Be sure to have proper protection against XSS.

### Summary:

- JWT is a standard for transmitting information between parties as a JSON object in a secure manner.
- It consists of three parts: header, payload, and signature.
- It's self-contained, compact, and works across different domains.
- The trade-offs include larger token size, the stateless nature, handling expiration