#seed 
upstream:

---

**links**: 

---

Brain Dump: 

--- 

# Understanding JSON Web Tokens (JWT)

## Introduction to JWT

JSON Web Tokens (JWT) are an open standard (RFC 7519) used for securely transmitting information between parties as a JSON object. This information can be verified and trusted because it is digitally signed. JWTs are commonly used for authentication and information exchange in web development.

## Structure of JWT

A JWT typically consists of three parts, each separated by a dot (`.`):

1. **Header**
2. **Payload**
3. **Signature**

### 1. Header

The header typically consists of two parts: the type of the token, which is JWT, and which of the [[Signing Algorithms]] being used, such as HMAC SHA256 or RSA.

**Example:**

```json
{
  "alg": "HS256",
  "typ": "JWT"
}
```

This JSON is Base64Url encoded to form the first part of the JWT.

### 2. Payload

The second part of the token is the payload, which contains the claims. Claims are statements about an entity (typically the user) and additional data. There are three types of claims: registered, public, and private claims.

- **Registered claims**: These are a set of predefined claims which are not mandatory but recommended to provide a set of useful, interoperable claims. Some of them are: `iss` (issuer), `exp` (expiration time), `sub` (subject), `aud` (audience), etc.

- **Public claims**: These can be defined at will by those using JWTs. To avoid collisions, they should be defined in the IANA JSON Web Token Registry or be defined as a URI that contains a collision-resistant namespace.

- **Private claims**: These are custom claims created to share information between parties that agree on using them and are neither registered nor public claims.

**Example:**

```json
{
  "sub": "1234567890",
  "name": "John Doe",
  "admin": true
}
```

This information is also Base64Url encoded to form the second part of the JWT.

### 3. Signature

To create the signature part, you have to take the encoded header, the encoded payload, a secret, the algorithm specified in the header, and sign that.

For example, if you are using the HMAC SHA256 algorithm, the signature will be created in the following way:

```python
HMACSHA256(
  base64UrlEncode(header) + "." +
  base64UrlEncode(payload),
  secret)
```

The signature is used to verify that the sender of the JWT is who it says it is and to ensure that the message wasn't changed along the way.

---

## How JWT Works

1. **User Authentication**: When the user successfully logs in using their credentials, a JWT is returned.

2. **JWT Storage**: The client application typically stores the JWT in local storage or session storage.

3. **Sending JWT in Requests**: Whenever the user wants to access a protected route or resource, the user agent should send the JWT, typically in the Authorization header using the Bearer schema.

4. **JWT Validation**: The server's protected routes will check for a valid JWT in the Authorization header, and if it's present, the user will be allowed to access protected resources.

5. **Expiration Handling**: JWTs are often short-lived, and new tokens can be issued by the server as needed, often by using a refresh token.

---

---

# JWT Security Considerations and Best Practices

## Security Aspects of JWT

While JWTs offer a compact and self-contained way to securely transmit information, it's essential to consider security aspects carefully:

### 1. **Use HTTPS**: 
To prevent token interception and man-in-the-middle attacks, always use JWTs in combination with HTTPS.

### 2. **JWT Storage**:
Where you store the JWT on the client side matters. Storing it in `localStorage` is convenient but susceptible to XSS attacks. A more secure approach is to use HttpOnly cookies, which are not accessible via JavaScript.

### 3. **Token Expiration**:
JWTs should have an expiration (`exp`) claim to reduce the damage if a token is stolen. Short-lived tokens are generally safer.

### 4. **Handling Token Expiration**:
Implement mechanisms on the client side to handle expired tokens, such as using refresh tokens to obtain new access tokens without forcing the user to log in again.

### 5. **Avoid Storing Sensitive Information**:
Do not store sensitive or personal identifiable information in JWT since it's easily decoded.

## Signing Algorithms

Choosing the right signing algorithm for JWT is crucial:

- **HMAC (HS256, HS384, HS512)**: Symmetric algorithms that use a single secret key for both signing and verification. Ensure the key is long and random to prevent brute-force attacks.
- **RSA (RS256, RS384, RS512)**: Asymmetric algorithms that use a private key for signing and a public key for verification. Suitable for scenarios where you want to share the key for verifying the token without exposing the key that signs it.

## Refresh Tokens

Refresh tokens are long-lived and used to request new access tokens. They should be stored securely on the server and sent to the client only when necessary. Implementing refresh token rotation and automatic blacklisting of used refresh tokens can enhance security.

## CSRF Protection with JWT

If you're using JWT in cookies for web applications, you still need to protect against CSRF attacks. One common approach is to use a CSRF token in addition to the JWT cookie.

## Best Practices

1. **Validate Input**: Always validate and sanitize user inputs to prevent injection attacks.
2. **Keep Libraries Updated**: Use trusted libraries for JWT and keep them updated to the latest version.
3. **Monitor and Log**: Implement monitoring and logging to detect unusual activities that could indicate a security issue.
4. **Regular Audits**: Conduct regular security audits and penetration testing to uncover and address potential vulnerabilities.

---

## Conclusion

JWT offers a robust and flexible method for authentication and information exchange in modern web applications. However, it's vital to implement them thoughtfully, considering the security implications and adhering to best practices to ensure the safety of your application and user data.

Remember, security in web applications is multi-faceted and requires a comprehensive approach beyond just handling tokens correctly. Regular updates, user education, secure coding practices, and a thorough understanding of web security are all crucial components of a secure web application.


