#bookmark 
upstream: [[Node.js]]

---

**video links**: 

---

## Understanding the `process` Module in `Node.js`

The `process` object is a global object in `Node.js` that provides information about, and control over, the current **`Node.js` process**. It can be accessed from anywhere in your `Node.js` application without having to import it. The `process` object is an instance of `EventEmitter` and emits several events you can listen to.

### Important Attributes

#### `process.env`
This is an object which holds the user environment. Most commonly, this is where you would read environment variables from, like `process.env.PORT` or `process.env.NODE_ENV`.
```js
console.log(process.env.PORT); // will print the value of PORT environment variable
console.log(process.env.NODE_ENV); // will print the value of NODE_ENV environment variable
```
*see [[Understanding Environment Variables in Node.js]] for more*

#### `process.argv`
This is an array containing the command-line arguments passed when the `Node.js` process was launched.
```js
// Run this script with extra arguments like `node script.js arg1 arg2`
console.log(process.argv); // will print ['/path/to/node', '/path/to/script.js', 'arg1', 'arg2']
```

#### `process.stdin`, `process.stdout`, `process.stderr`
These properties are streams that represent the standard input, standard output, and standard error, respectively.
```js
process.stdin.on('data', (data) => {
  console.log(`You typed: ${data}`);
});

process.stdout.write('Hello, world!\n'); // prints "Hello, world!" to the console

process.stderr.write('An error occurred\n'); // prints "An error occurred" to the error console
```

#### `process.pid`
The **PID** (Process ID) of the process.
```js
console.log(`Process ID: ${process.pid}`); // will print the process ID
```

#### `process.platform`
The platform on which the `Node.js` process is running, e.g., `'darwin'`, `'win32`', etc.
```js
console.log(`Platform: ${process.platform}`); // will print the platform the process is running on
```

#### `process.version`
The version of `Node.js`
```js
console.log(`Node.js version: ${process.version}`); // will print the version of Node.js
```

### Important Methods

- `process.cwd()`: This method returns the current working directory of the process.

- `process.chdir(directory)`: This method changes the current working directory of the process.

- `process.exit([code])`: This method instructs Node.js to terminate the process synchronously with an exit status of `code`. If `code` is omitted, it defaults to `0`.

- `process.nextTick(callback[, ...args])`: This method adds the callback to the "next tick queue" which will execute after the current operation completes, regardless of the current phase of the event loop.

- `process.kill(pid[, signal])`: This method sends a signal to a process (normally causing termination).

- `process.uptime()`: This method returns the number of seconds the process has been running.

The `process` module is essential for gaining insights into the current Node.js process and for controlling it. It's especially important for managing environment-specific configuration (like database credentials), handling command-line arguments, managing asynchronous tasks, or handling process signals.




