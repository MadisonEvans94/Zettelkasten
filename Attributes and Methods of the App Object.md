#incubator 
###### upstream: 

### Core Thought: 

When you initialize an Express.js application using `const app = express();`, you instantiate an Express application object that has a variety of built-in properties and methods which provide the functionality for building a web application. 

#### Properties

1. `app.locals`: an object upon which you can define variables that are scoped to the application level. These variables, or "application-level locals", are available across all the views within your application. *This is conceptually similar to [[useContext]]* 

2. `app.mountpath`: A property that contains the path pattern(s) on which a sub-app was mounted. *see [[sub-apps]] for more details*

3. `app.parent`: A reference to the parent application when an app is mounted. 

#### Methods

1. `app.use([path], function)`: Mounts the specified middleware function or functions at the specified path. If path is not specified, it defaults to "/" and matches all routes.
2. `app.listen(path, [callback])`: Starts a UNIX socket and listens for connections on the given path.
3. `app.listen(port, [hostname], [backlog], [callback])`: Starts a TCP server listening for connections on the given port and hostname.
4. `app.all(path, callback[, callback ...])`: This method routes HTTP requests to the specified path with the specified callback functions for all HTTP methods.
5. `app.get(path, callback[, callback ...])`: Routes HTTP GET requests to the specified path with the specified callback functions.
6. `app.post(path, callback[, callback ...])`: Routes HTTP POST requests to the specified path with the specified callback functions.
7. `app.put(path, callback[, callback ...])`: Routes HTTP PUT requests to the specified path with the specified callback functions.
8. `app.delete(path, callback[, callback ...])`: Routes HTTP DELETE requests to the specified path with the specified callback functions.
9. `app.disable(name)`: Sets the boolean setting name to false.
10. `app.disabled(name)`: Returns true if the setting name is disabled (false), or false otherwise.
11. `app.enable(name)`: Sets the setting name to true.
12. `app.enabled(name)`: Returns true if the setting name is enabled (true), or false otherwise.
13. `app.engine(ext, callback)`: Registers the given template engine callback as ext.
14. `app.get(name)`: Returns the value of name app setting.
15. `app.param([name], callback)`: Adds callback triggers to route parameters, where name is the name of the parameter or an array of them.
16. `app.path()`: Returns the canonical route path for the application, a string.
17. `app.render(view, [locals], callback)`: Renders a view and sends the rendered HTML string to the client.
18. `app.route(path)`: Returns an instance of a single route, which you can then use to handle HTTP verbs with optional middleware.
19. `app.set(name, value)`: Assigns setting name to value.

*These are the main properties and methods that come out-of-the-box when you initialize an Express.js application. However, keep in mind that Express.js is a living project and may evolve with new methods or properties over time*
