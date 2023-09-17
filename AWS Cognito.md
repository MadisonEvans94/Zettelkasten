#seed 
upstream: [[AWS]]

---

**video links**: 

---

## Heading 1:

While some organizations may opt to implement their own authentication and password hashing solutions using libraries like bcrypt, many are moving towards managed services that handle these tasks due to their robust security, scalability, and reduced maintenance overhead.

For AWS specifically, the managed service that handles user authentication is Amazon Cognito. It provides a secure and scalable user directory that can scale to hundreds of millions of users. Amazon Cognito takes care of the secure storage and hashing of user passwords, sign-up, sign-in, and access control, so you don't have to.

With Amazon Cognito, you can also integrate with social identity providers such as Google, Facebook, and Amazon, and enterprise identity providers via SAML 2.0. It also supports multi-factor authentication and encryption of data-at-rest and in-transit.

That being said, the choice between implementing your own user authentication and password management or using a managed service like Amazon Cognito depends on various factors like the complexity of your application, your security requirements, the size of your user base, your budget, and your team's expertise.

If you opt to go with a managed service like Amazon Cognito, ensure you understand its cost model, as well as any potential limitations or caveats that might come with it. If you decide to handle user authentication and password hashing on your own, make sure you adhere to best practices, such as securely hashing and salting passwords (as bcrypt does), never storing plaintext passwords, and using secure, up-to-date libraries and algorithms.

### Heading 2: 
#### Heading 3: 


