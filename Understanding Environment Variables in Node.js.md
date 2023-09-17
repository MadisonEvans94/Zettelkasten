#bookmark

## Introduction

Environment variables are a fundamental part of developing with Node.js, allowing your app to behave differently based on the environment you want them to run in. They are part of the environment in which a process runs. For example, if you are running a test server and a production server, you might want them to connect to different databases, which you can manage with environment variables.

## Environment Variables in Node.js

Environment variables in Node.js are accessible through the global `process.env` object. They are dynamically-named variables in your environment that can be used to set configurable values for your software.

The `process.env` object is special in Node.js. It holds a reference to your environment's variables as key-value pairs. When you run a process from the command line, `process.env` returns an object containing all environment variables.

Here is an example of how to access environment variables:

```javascript
console.log(process.env.PATH);
```

This will log the value of the `PATH` environment variable, which is a system environment variable that tells the system where to look for executable files.

You can also set environment variables directly in your environment, in your hosting provider's settings, or even in your CI/CD pipeline.

## Setting Environment Variables

Setting environment variables can differ between OSes. Here's how you can do it:

- On Unix-based systems (like Linux and MacOS), you can use the `export` keyword:

    ```bash
    export VAR_NAME="My Value"
    ```

- On Windows, you can use the `set` keyword:

    ```bash
    set VAR_NAME="My Value"
    ```

In your Node.js code, you can access this value with `process.env.VAR_NAME`.

## `.env` Files and `dotenv`

Storing configuration in the environment is based on The Twelve-Factor App methodology. As a part of this methodology, the `dotenv` module was developed. It allows you to load environment variables from a `.env` file into `process.env`. This makes managing configurations for different environments very easy.

Here's how you use it:

1. Install `dotenv` as a development dependency:

    ```bash
    npm install dotenv --save
    ```

2. Create a `.env` file in the root of your project:

    ```text
    VAR_NAME=My Value
    ```

3. As early as possible in your application, require and configure `dotenv`:

    ```javascript
    require('dotenv').config()
    ```

You can now access `VAR_NAME` from `process.env.VAR_NAME` in your code.

## Understanding the Node.js Runtime Environment

The Node.js runtime environment is a platform for running JavaScript. Node.js is built on Chrome's V8 JavaScript engine, and it's mainly used to create web servers, tools, or scripts. 

Node.js is asynchronous and event-driven in nature, meaning it follows non-blocking I/O model that makes it lightweight and efficient. This makes it suitable for real-time applications that run across distributed systems.

Node.js includes a REPL environment for interactive testing and debugging, a package manager (npm) for installing and managing libraries, and it supports a range of core modules such as file system, streams, HTTP, etc.

When a Node.js process is launched, it sets up the environment for the JavaScript code to run in, preparing things like environment variables (`process.env`), command-line arguments (`process.argv`), and handling uncaught exceptions (`process.on('uncaughtException', func)`). 

The `process` object is a global that provides information about, and control over, the current Node.js process. It can be accessed from anywhere in your Node.js application without having to import it.

 

Note: Although the `process` object is global, it's not part of the JavaScript language but specific to the environment where your code is running. Thus, `process.env` or `process.argv` won't work in a browser's JavaScript runtime.
