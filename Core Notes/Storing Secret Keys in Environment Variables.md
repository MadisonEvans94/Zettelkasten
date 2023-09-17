#incubator 

upstream: [[Security Best Practices in Express]], [[Web Development]]

### Why Do It? 

Storing application configuration and secrets such as **database passwords**, **API keys**, or **JWT secret keys** directly in your code is a security risk and is not a best practice. Storing these sensitive details in environment variables is one way to mitigate this risk. 

### What are Environment Variables?

**Environment variables** are dynamic named values that can affect the way running processes will behave on a computer. They exist in every operating system, and their types or usage might vary, but their purpose is to define certain data for the system's usage. 

For a `Node.js` application, we can use environment variables to allow our application to adjust its behavior based on the running environment. This way, you could have certain behavior for your development environment and different behavior for your production environment.

### Why Use Environment Variables?

The main reasons to use environment variables are:

1. **Security**: Store sensitive data securely and separately from the code base.
2. **Scalability**: Makes the code more flexible and easy to update or change.
3. **Separation of Concerns**: Keeps business logic separate from configuration and environment specifics.

### How to Use Environment Variables in a Node.js application?

The **Node.js Runtime** has a global object `process.env` that allows you to access environment variables. For example, `process.env.PORT` could be used to determine the port on which your server runs.

*However*, hardcoding secret keys even in environment variables on your server is not a good practice, especially if your code is version-controlled in a public repository.

### Using the `dotenv` Library

A common approach to handling this is to use a package like `dotenv`, which allows you to easily configure environment variables in a file that you can add to your `.gitignore`, keeping the secrets out of your version-controlled codebase.

Here are the steps to do it:

#### 1. **Install `dotenv`**: 

First, install the `dotenv` package in your Node.js project. Run the following command in your terminal:

```bash
npm install dotenv
```

#### 2. **Create a `.env` file**: 

In the root of your project, create a new file called `.env`. Here, you'll store your environment variables. For example:

```dotenv
SECRET_KEY=mySuperSecretKey
DB_CONNECTION_STRING=myConnectionString
PORT=4000
```

Each line in the `.env` file should represent a key-value pair.

#### 3. **Add `.env` to `.gitignore`**: 

You should add your `.env` file to your `.gitignore` file. This ensures that the `.env` file, containing all your secrets, will not be tracked by Git and accidentally get pushed to a remote repository.

#### 4. **Access the Variables**: 

Now, you'll need to configure `dotenv` at the very start of your application:

```javascript
import dotenv from 'dotenv';
dotenv.config();
```

After you've done this, you can access any of the variables you set in your `.env` file:

```javascript
console.log(process.env.SECRET_KEY);
console.log(process.env.DB_CONNECTION_STRING);
console.log(process.env.PORT);
```


### Environment Variables in Production

In a production environment, you would not use a `.env` file. Instead, the environment variables would be set directly on the production server. This is often done through the interface of the hosting service (like AWS, Heroku, Vercel, etc.) you're using. 

See [[Understanding Environment Variables in AWS]] for more details 

