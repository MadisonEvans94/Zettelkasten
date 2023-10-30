#seed 
upstream:

---

**links**: 

---

Flask's app object, an instance of the `Flask` class, plays a central role in every Flask application. It's responsible for handling route definitions, configuration, registering blueprints, and much more. Below are some of the most commonly used attributes and methods associated with this object:

### Attributes

1. **`app.config`**: 
   - A dictionary-like object containing configuration parameters.
   - Example:
     ```python
     app.config["DEBUG"] = True
     ```

2. **`app.logger`**: 
   - Provides a logger that's configured to work with Flask's setup.
   - Example:
     ```python
     app.logger.info('Info message')
     ```

3. **`app.root_path`**:
   - Gives the path to the root directory of the application.

4. **`app.static_folder`**:
   - The folder with static files that should be served at `/static`.

5. **`app.template_folder`**:
   - The folder where Flask will look for Jinja2 templates.

6. **`app.url_map`**:
   - Contains the map of routes and the associated view functions.

### Methods

1. **`app.route(rule, **options)`**:
   - A decorator to bind a URL rule to a function.
   - `rule`: The URL rule as a string.
   - `options`: Options for the route, like methods=['GET', 'POST'].
   - Example:
     ```python
     @app.route('/')
     def home():
         return "Home Page"
     ```

2. **`app.add_url_rule(rule, endpoint=None, view_func=None, **options)`**:
   - Connects a URL rule to a function without the decorator.
   - `rule`: The URL pattern.
   - `endpoint`: The name of the endpoint. If not provided, the function name is used.
   - `view_func`: The function to call when the route is accessed.
   - Example:
     ```python
     def about():
         return "About Page"
     app.add_url_rule('/about', 'about', about)
     ```

3. **`app.before_request(func)`**:
   - Registers a function to run before each request.
   - Example:
     ```python
     @app.before_request
     def log_request():
         app.logger.info('Request made')
     ```

4. **`app.after_request(func)`**:
   - Registers a function to run after each request. The function must return a response object.
   - Example:
     ```python
     @app.after_request
     def add_header(response):
         response.headers['X-Custom-Header'] = 'My custom header value'
         return response
     ```

5. **`app.errorhandler(code)`**:
   - Provides a way to customize error responses.
   - Example:
     ```python
     @app.errorhandler(404)
     def not_found(error):
         return "Page Not Found!", 404
     ```

6. **`app.register_blueprint(blueprint, **options)`**:
   - Registers a blueprint with the app.
   - Example:
     ```python
     from mymodule import my_blueprint
     app.register_blueprint(my_blueprint)
     ```

7. **`app.run(host=None, port=None, debug=None, **options)`**:
   - Runs the application on a local development server.
   - `host`: The hostname to listen on. Defaults to `127.0.0.1`.
   - `port`: The port of the webserver. Defaults to `5000`.
   - `debug`: If given, enables or disables debug mode.
   - Example:
     ```python
     app.run(port=8000, debug=True)
     ```

This document outlines only a subset of the available attributes and methods of the Flask app object. For a comprehensive overview and details on usage, always refer to the [official Flask documentation](https://flask.palletsprojects.com/).

---




