#seed 
upstream: [[Web Development]]

---

**video links**: 

---

# Brain Dump: 


--- 

## Introduction to PHP <a name="introduction-to-php"></a>

**PHP (Hypertext Preprocessor)** is a popular general-purpose scripting language that is especially suited to web development. It was created by Rasmus Lerdorf in 1994.

PHP is a server-side language, which means it runs on the server and generates HTML which is sent to the client. PHP can be embedded within HTML, and it can also interact with various database systems.

### Key Features of PHP:

- **Server-side scripting language:** PHP code is executed on the server, and the result is returned to the browser as plain HTML.

- **Cross-platform:** PHP runs on various platforms (Windows, Linux, Unix, Mac OS X, etc.).

- **Database Compatibility:** PHP supports a wide range of databases, including MySQL, PostgreSQL, Oracle, Sybase, etc.

- **Embedded:** PHP code can be embedded within HTML code.

---

## PHP Syntax <a name="php-syntax"></a>


### Basic Syntax <a name="basic-syntax"></a>

PHP scripts start with `<?php` and end with `?>`. A PHP file normally contains HTML tags, and some PHP scripting code. Here is an example of a simple PHP file, with a PHP script that uses a built-in PHP function "echo" to output the text "Hello, World!" on a web page:

```php
<!DOCTYPE html>
<html>
	<body>
		<h1>My first PHP page</h1>
		<?php
			echo "Hello, World!";
		?>
	</body>
</html>
```

---

### Variables <a name="variables"></a>

In PHP, a variable starts with the `$` sign, followed by the name of the variable. PHP variables are case-sensitive.

```php
<?php
$txt = "Hello, World!";
$x = 5;
$y = 10.5;
```

---

### Data Types <a name="data-types"></a>

PHP supports the following data types:

- String
- Integer
- Float (floating point numbers - also called double)
- Boolean
- Array
- Object
- NULL

### Control Structures <a name="control-structures"></a>

PHP supports the following control structures:

#### If...Else...Elseif

```php
$weather = "sunny";

if ($weather == "sunny") {
    echo "Let's go to the park.";
} elseif ($weather == "rainy") {
    echo "Let's stay home.";
} else {
    echo "Let's go to the cinema.";
}
```

#### Switch

```php
$fruit = "apple";

switch ($fruit) {
    case "apple":
        echo "Your favorite fruit is apple!";
        break;
    case "banana":
        echo "Your favorite fruit is banana!";
        break;
    default:
        echo "Your favorite fruit is neither apple nor banana!";
}
```

#### While Loop

```php
$x = 1;

while($x <= 5) {
    echo "The number is: $x <br>";
    $x++;
}
```

#### Do...While Loop

```php
$x = 1;

do {
    echo "The number is: $x <br>";
    $x++;
} while ($x <= 5);
```

#### For Loop

```php
for ($x = 0; $x <= 10; $x++) {
    echo "The number is: $x <br>";
}
```

#### Foreach Loop

```php
$colors = array("red", "green", "blue", "yellow");

foreach ($colors as $value) {
    echo "$value <br>";
}
```



### Understanding `echo` and `return` in PHP <a name="echo-and-return"></a>

In PHP, `echo` and `return` are used for different purposes. Understanding these differences is crucial for writing effective PHP code.

#### The `echo` Statement <a name="echo-statement"></a>

The `echo` statement is a language construct in PHP that outputs one or more strings. It's typically used to output data to the web browser. Here's an example:

```php
echo "Hello, World!";
```

In this example, the `echo` statement outputs the string "Hello, World!" to the web browser.


#### The `return` Statement <a name="return-statement"></a>

The `return` statement is used within a function to stop the execution of the function and send back a value to the calling code. If the `return` statement is not within a function, it will stop the execution of the script. Here's an example:

```php
function add($x, $y) {
    $sum = $x + $y;
    return $sum;
}

$result = add(5, 10);
echo $result; // Outputs: 15
```

In this example, the `add` function returns the sum of its parameters, and this value is then outputted using `echo`.

In summary, `echo` is used for outputting data to the browser, and `return` is used to stop the execution of a function and send a value back to the calling code.

> so let's say I'm writing PHP script within an div element in html. What would be the difference if I returned a string vs echoed a string?

When you're writing PHP script within an HTML document, `echo` is typically what you would use to output a string into the HTML document. This is because `echo` sends the output to the browser, which then renders it as part of the HTML.

Here's an example:

```php
<div>
    <?php
        echo "Hello, World!";
    ?>
</div>
```

In this case, "Hello, World!" would be displayed on the webpage within the `<div>` element.

On the other hand, `return` in PHP is used to send a value back to the calling code and stop the execution of the current function. If you use `return` in a PHP script within an HTML document (outside of a function), it will not output anything to the HTML. Instead, it will stop the execution of the script.

Here's an example:

```php
<div>
    <?php
        return "Hello, World!";
    ?>
</div>
```

In this case, "Hello, World!" would not be displayed on the webpage. The PHP script would simply stop executing at the `return` statement.

So, in the context of a PHP script within an HTML document, you would typically use `echo` to output a string to the HTML. The `return` statement is more commonly used within functions to send a value back to the calling code.

### Functions <a name="functions"></a>

A function is a block of statements that can be used repeatedly in a program. A function will not execute immediately when a page loads. A function will be executed by a call to the function.

```php
<?php
function writeMsg() {
  echo "Hello, World!";
}

writeMsg(); // call the function
?>
```

---

### Comparison with Other Languages <a name="comparison-with-other-languages"></a>

- **JavaScript:** PHP is executed on the server, while JavaScript is executed on the client. PHP is a server-side scripting language, while JavaScript is mainly a client-side scripting language.

- **Python:** PHP is traditionally used as a server-side scripting language for web development, while Python is valued for its dynamics, availability, and simplicity. Despite its syntax, PHP is not as elegant as Python.

- **Java:** PHP is an interpreted language, while Java is a compiled language. PHP is mainly used for web development, while Java is used in a wide variety of computing platforms from embedded devices and mobile phones to enterprise servers and supercomputers.

---

## Introduction to Laravel <a name="introduction-to-laravel"></a>

**Laravel** is a web application framework with expressive, elegant syntax. It attempts to take the pain out of development by easing common tasks used in the majority of web projects, such as routing, sessions, caching, and authentication.

### Key Features of Laravel:

#### MVC Architecture
Laravel follows the Model-View-Controller design pattern, ensuring clarity between logic and presentation. This design pattern allows for efficient code organization and separation of concerns. 

> see [[Understanding MVC Architecture with a Web Example]]


#### Eloquent ORM
Laravel includes Eloquent, an advanced Object-Relational Mapper (ORM) for working with your database. Each database table has a corresponding "Model" that allows you to interact with that table.

#### Security
Laravel takes care of the security within its framework. It uses hashed and salted password mechanisms so the password would never be saved as plain text in a database. It also uses "csrf" token to prevent cross-site request forgery.

#### Artisan
Laravel includes Artisan, a **CLI** tool that assists you in building your application by providing helpful commands for common tasks.

---

## Comparison with Express.js <a name="comparison-with-expressjs"></a>

Express.js is a minimal and flexible Node.js web application framework that provides a robust set of features for web and mobile applications. It is part of the MEAN (MongoDB, Express.js, Angular.js, Node.js) stack, which is a full-stack JavaScript solution that helps you build fast, robust, and maintainable production web applications.

>see [[express and laravel side by side]]

### Key Differences:

- **Language:** Express.js is a JavaScript framework, while Laravel is a PHP framework. This means that if you're more comfortable with JavaScript, you might find Express.js easier to use.

- **Database Interaction:** Express.js uses different libraries like Mongoose for interacting with databases, while Laravel uses Eloquent ORM.

- **Template Engine:** Express.js uses Jade as a template engine, while Laravel uses Blade.

- **Middleware:** Both Express.js and Laravel use middleware, but the way they handle it is different. Express.js uses middleware for every route by default, while Laravel has a more flexible and intuitive way to handle middleware.
> see the middleware section for more on this 

- **Learning Curve:** Laravel, with its many out-of-the-box features, might have a steeper learning curve compared to Express.js, which is more minimalistic.


Sure, let's dive into middleware in Laravel.

---

## Middleware in Laravel <a name="middleware-in-laravel"></a>

Middleware provides a convenient mechanism for filtering HTTP requests entering your application. For example, Laravel includes a middleware that verifies the user of your application is authenticated. If the user is not authenticated, the middleware will redirect the user to the login screen. However, if the user is authenticated, the middleware will allow the request to proceed further into the application.

---

### Understanding Middleware <a name="understanding-middleware"></a>

In Laravel, a middleware is a bridge or connector that helps to control the HTTP request and response process. It works between the request and response of our application. It provides a way to filter out the HTTP requests on your site.

---

### Creating Middleware <a name="creating-middleware"></a>

You can create a new middleware by using the `make:middleware` Artisan command:

```bash
php artisan make:middleware CheckAge
```

This command will place a new `CheckAge` class within your `app/Http/Middleware` directory. In this middleware, we will only allow access to the route if the supplied age is greater than 200. Otherwise, we will redirect the users back to the home URI:

```php
<?php

namespace App\Http\Middleware;

use Closure;

class CheckAge
{
    public function handle($request, Closure $next)
    {
        if ($request->age <= 200) {
            return redirect('home');
        }

        return $next($request);
    }
}
```

---

### Registering Middleware <a name="registering-middleware"></a>

After creating the middleware, you need to register it in the `app/Http/Kernel.php` file. There are two types of middleware in Laravel:

- **Global Middleware:** If you want a middleware to be run during every HTTP request to your application, list the middleware class in the `$middleware` property of your `app/Http/Kernel.php` class.

- **Route Middleware:** If you want to assign middleware to specific routes, you should assign the middleware a key in your `app/Http/Kernel.php` file. By default, the `$routeMiddleware` property of this class contains entries for the middleware included with Laravel.

---

### Middleware Groups <a name="middleware-groups"></a>

Sometimes you may want to group several middleware under a single key to make them easier to assign to routes. You can do this using the `$middlewareGroups` property in your `app/Http/Kernel.php` file.

---

### Common Middleware Packages <a name="common-middleware-packages"></a>

There are several commonly used middleware packages in Laravel:

- **Laravel CORS (Cross-Origin Resource Sharing):** Handles CORS pre-flight OPTIONS requests and adds the necessary headers to responses.

- **Laravel Permission:** Allows you to manage user permissions and roles in a database.

- **Laravel Sanctum:** Provides a featherweight authentication system for SPAs (single page applications), mobile applications, and simple, token based APIs.


Sure, let's dive into PHP syntax.

---

## PHP and Laravel Best Practices <a name="php-laravel-best-practices"></a>

Following best practices for PHP and Laravel can help you write cleaner, more efficient, and more maintainable code.

### Table of Contents

1. [PHP Best Practices](#php-best-practices)
2. [Laravel Best Practices](#laravel-best-practices)

---

### PHP Best Practices <a name="php-best-practices"></a>

- **Use the Latest PHP Version:** Always use the latest stable version of PHP. Each new version typically comes with performance improvements, new features, and bug fixes.

- **Follow PSR Standards:** The PHP Framework Interop Group (PHP-FIG) has proposed a number of style guides and standards that you should follow to write clean and easy-to-read code.

- **Avoid Using `@` Error Control Operator:** The `@` operator can hide error messages that you need to know about.

- **Use OOP:** Object-Oriented Programming (OOP) can help you write more modular and reusable code.

- **Use Composer:** Composer is a tool for dependency management in PHP. It allows you to declare the libraries your project depends on and it will manage (install/update) them for you.

- **Avoid Using `mysql_*` Functions:** The `mysql_*` functions are deprecated as of PHP 5.5.0. Use `mysqli` or `PDO_MySQL` instead.

---

### Laravel Best Practices <a name="laravel-best-practices"></a>

- **Use Eloquent ORM and Relationships:** Eloquent ORM provides an easy way to interact with your database using object-oriented syntax.

- **Use Artisan Command Line Tool:** Artisan provides a number of helpful commands for common tasks.

- **Use Migration for Database:** Migrations are like version control for your database, allowing your team to modify and share the application's database schema.

- **Follow MVC Pattern:** Laravel is an MVC framework, so use controllers for your application logic, models for your data and relationships, and views for your UI.

- **Use Route Groups:** Route groups allow you to share route attributes, such as middleware or namespaces, across a large number of routes without needing to define those attributes on each individual route.

- **Use Laravel Mix for Assets:** Laravel Mix provides a fluent API for defining Webpack build steps for your application.

- **Use Validation Requests:** Validation requests are the way Laravel handles validation. By using validation requests, you can keep your controllers slim.

- **Use Configuration Files and Environment Variables:** Laravel's configuration files are stored in the `config` directory. Use environment variables in your configuration files to keep sensitive information out of your code.


Sure, let's dive into setting up a basic Laravel application that can work with a React frontend.

---

## Setting Up a Basic Laravel Application <a name="setting-up-laravel"></a>

This section will guide you through the process of setting up a basic Laravel application. We'll also cover how to set up Laravel to work with a React frontend.

### Table of Contents

1. [Installing Laravel](#installing-laravel)
2. [Setting Up the Database](#setting-up-database)
3. [Creating Routes, Controllers, and Models](#creating-routes-controllers-models)
4. [Setting Up Laravel to Work with React](#setting-up-laravel-react)

---

### Installing Laravel <a name="installing-laravel"></a>

Before installing Laravel, make sure you have Composer installed on your machine. You can install Laravel by issuing the Composer `create-project` command in your terminal:

```bash
composer create-project --prefer-dist laravel/laravel blog
```

This command will create a new Laravel project in a directory named `blog`.

---

### Setting Up the Database <a name="setting-up-database"></a>

After installing Laravel, the next step is to set up the database. You can do this by editing the `.env` file in the root of your Laravel project:

```bash
DB_CONNECTION=mysql
DB_HOST=127.0.0.1
DB_PORT=3306
DB_DATABASE=your_database_name
DB_USERNAME=your_username
DB_PASSWORD=your_password
```

---

### Creating Routes, Controllers, and Models <a name="creating-routes-controllers-models"></a>

In Laravel, you can define all of the routes for your application in the `routes/web.php` file. Here's an example of a basic route that returns a view:

```php
Route::get('/', function () {
    return view('welcome');
});
```

You can generate a controller using the `make:controller` Artisan command:

```bash
php artisan make:controller PostController
```

In Laravel, each database table has a corresponding "Model" that is used to interact with that table. You can generate a model using the `make:model` Artisan command:

```bash
php artisan make:model Post
```

---

### Setting Up Laravel to Work with React <a name="setting-up-laravel-react"></a>

Laravel comes with a built-in support for React. You can create a new Laravel project with React set up out of the box by using the `--preset` option of the `make:ui` Artisan command:

```bash
php artisan ui react
```

This command will create a sample React component in the `resources/js/components` directory. The `resources/js/app.js` file will load this component. You can compile your React components using Laravel Mix:

```bash
npm install
npm run dev
```

---

In the next section, we will cover how to create APIs in Laravel.

---

Let me know if you're ready to proceed with the next sections.