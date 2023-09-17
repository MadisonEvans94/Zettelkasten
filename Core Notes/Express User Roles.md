[Create User Roles in NodeJS and ReactJS - Tutorial](https://www.youtube.com/watch?v=YLihWZwLaGU&ab_channel=PedroTech)

#incubator 
###### upstream: 

### Creating User Roles in Express

Creating user roles is an important part of building a secure and efficient Express.js application. Here's a step-by-step guide on how to implement it:

### Step-by-step

#### Step 1: User Model

The first step is to define your user model. A common practice is to add a `role` field to your user schema. This could be a string or an integer, depending on whether you prefer to use role names or role levels. 

```javascript
import mongoose from 'mongoose';

const userSchema = new mongoose.Schema({
  username: String,
  password: String,
  role: {
    type: String,
    enum: ['user', 'admin', 'superadmin'],
    default: 'user'
  }
});

const User = mongoose.model('User', userSchema);

export default User;
```

In this example, we've defined three roles: `user`, `admin`, and `superadmin`.

#### Step 2: Authentication

Before you can authorize based on roles, you need to have authentication in place. This can be done through various strategies, such as using **JWT** [[JWT (JSON Web Token)]], **session cookies**, or **OAuth**.

Here is an example of how to encode user role into a JWT:

```javascript
import jwt from 'jsonwebtoken';

function generateToken(user) {
  return jwt.sign({
    userId: user._id,
    role: user.role
  }, 'YOUR_SECRET_KEY');
}
```
*Keep in mind that you need to run `npm i jsonwebtoken` first. To learn more about the specific implementation of jwt object, go to [[The jsonwebtoken Library]]...*

When the user logs in or signs up, the server would generate this token and send it back to the client. The client would then include this token in the `Authorization` header for subsequent requests.

#### Step 3: Middleware for Authorization

Now that we have the user's role in each request (via JWT, for example), we can write middleware to check if a user is authorized to perform certain operations. 

```javascript
function authorize(roles = []) {
  return (req, res, next) => {
    // assuming req.user is already set by a previous authentication middleware
    if (!roles.includes(req.user.role)) {
      // user's role is not authorized
      return res.status(401).json({ message: 'Unauthorized' });
    }

    // authentication and authorization successful
    next();
  }
}
```

#### Step 4: Apply Middleware to Routes

Finally, you can apply the `authorize` middleware to your routes:

```javascript
app.post('/admin', authorize('admin'), (req, res) => {
  res.json({ message: 'Hello admin!' });
});

app.post('/superadmin', authorize('superadmin'), (req, res) => {
  res.json({ message: 'Hello superadmin!' });
});
```

This way, only admin users can access the '/admin' route and only superadmin users can access the '/superadmin' route.

### Summary:

- Add a `role` field to your User model.
- Implement authentication and include the user's role in the auth token.
- Write middleware to authorize based on roles.
- Apply the middleware to your routes.

Remember, this is a simplified example and in a production application, you'll want to consider other factors such as securely storing your secret key, encrypting the user's password, handling different error types, etc.