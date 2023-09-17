#evergreen1 
upstream:

---

**video links**: 
- [Learn Express Middleware in 14 Minutes](https://youtu.be/lY6icfhap2o)

---


[[Middleware]] functions are essential in **`Express.js`** because they have access to the **request object** (`req`), the **response object** (`res`), and the **next function** in the application’s request-response cycle, which is typically named `next`. These functions can execute any code, make changes to the request and response objects, end the request-response cycle, or call the next function in the stack.

### Examples

#### 1. Body Parser Middleware (`body-parser`): 
*The ability for express to parse json has to be manually added, so it's a good idea to go ahead and include that middleware during setup workflow* 

It's used to parse incoming request bodies in a middleware before your handlers, available under the `req.body` property. As of Express 4.16.0, body parsing middleware is included with Express. You don't need to use the `body-parser` module separately. [docs](https://expressjs.com/en/resources/middleware/body-parser.html)

```js
app.use(express.json()); // for parsing application/json
app.use(express.urlencoded({ extended: true })); // for parsing application/x-www-form-urlencoded
```

#### 2. CORS Middleware (`cors`): 

CORS, or **Cross-Origin Resource Sharing**, is a mechanism that uses additional HTTP headers to allow a user agent (like a web browser) to gain permission to access resources from a server at a different origin. By default, web browsers prohibit web pages from making requests to a different domain than the one the web page came from. [docs](https://expressjs.com/en/resources/middleware/cors.html)

...first install `cors`: 
```bash
npm install cors
```

...then add to code 
```js
import cors from 'cors';
app.use(cors());
```

by default, invoking `cors` within argument allows **all** url paths. To indicate only specific routes, use the following syntax: 

```javascript
const corsOptions = {
  origin: 'http://example.com',
  optionsSuccessStatus: 200 // some legacy browsers (IE11, various SmartTVs) choke on 204
}

app.use(cors(corsOptions))
```

#### 3. Express Session (`express-session`):

It's used to manage user sessions in your applications. [docs](https://www.npmjs.com/package/express-session)
```js
import session from 'express-session';
app.use(session({ secret: 'mysecret', resave: true, saveUninitialized: true }));
```

*see [[Sessions in Express]] for more details*

#### 4.  Passport (`Passport.js`): 
It's an authentication middleware for `Node.js`. Extremely flexible and modular, Passport can be unobtrusively dropped into any Express-based web application. [docs](https://www.passportjs.org/docs/)
```js
import passport from 'passport';
app.use(passport.initialize());
app.use(passport.session());
```

*see [[Passport.js]] for more*

#### 5. Multer (`multer`): 

It's a middleware for handling `multipart/form-data`, which is primarily used for **uploading files**. [docs](https://expressjs.com/en/resources/middleware/multer.html)

```js
import multer from 'multer';
app.use(multer({ dest: './uploads/'}).single('avatar'));
```

*see [[Multer vs Node.js process module]] for more*

#### 6. `bcrypt`: 

If you're using `bcrypt` middleware to hash the user's **password** before it is saved in the database, you will need to modify your user model and use `Sequelize` hooks to do this.

First, install `bcrypt`:

```bash
npm install bcrypt
```

Then you can modify your `models/user.js` file like this:

```javascript
import { DataTypes } from "sequelize";
import bcrypt from 'bcrypt';

const User = (sequelize, DataTypes) => {
	const UserModel = sequelize.define("User", {
		username: DataTypes.STRING,
		birthday: DataTypes.DATE,
		password: {
			type: DataTypes.STRING,
			validate: {
				notEmpty: true,
			}
		}
	});

	UserModel.beforeCreate(async (user, options) => {
		const salt = await bcrypt.genSalt();
		user.password = await bcrypt.hash(user.password, salt);
	});

	return UserModel;
};

export default User;
```

In this code, we added a new `password` field to our user model and implemented a Sequelize `beforeCreate` hook. This hook will automatically hash the password using `bcrypt` before any new user is created.

Additionally, you may want to handle the password comparison for user authentication. This could be added to the model instance methods like so:

```javascript
UserModel.prototype.isValidPassword = async function(password) {
    return await bcrypt.compare(password, this.password);
};
```

This function allows you to compare a plain-text password (`password`) with the hashed password stored in the database (`this.password`).

*Note* 

If you're on a tight budget and your application is small and not very complex, then using `bcrypt` for password hashing in your Express server can be a good choice. It's a reliable library that follows the best practices for password storage, including automatic salting and multiple hashing rounds.

However, there are a few considerations you should be aware of:

1. **Security**: Even if your application is small and simple, security should always be a priority, especially when dealing with user passwords. You'll need to ensure your implementation is secure and that you're following best practices for handling sensitive user data. Mistakes in implementing authentication can lead to serious security vulnerabilities.

2. **Maintenance**: Building your own authentication means you'll be responsible for maintaining that part of your codebase, which can become complex and time-consuming as your app grows and needs change.

3. **Scaling**: If your application grows, you may need to consider more robust and scalable solutions. 

4. **Features**: Managed services like [[AWS Cognito]] offer many additional features like multi-factor authentication, integration with social identity providers, and access control, which you would have to implement yourself if you need them.


