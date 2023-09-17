#seed 
upstream:  [[Ruby]]

### Introduction

Ruby on Rails (often just called Rails) is an open-source web application framework written in Ruby. It follows the Model-View-Controller (MVC) architectural pattern and provides a default structure for databases, web services, and web pages. Rails encourages the use of well-known software engineering patterns such as Convention over Configuration (CoC), Don't Repeat Yourself (DRY), and Active Record.

### Ruby

[[Ruby]] is a high-level, interpreted, and object-oriented programming language developed by Yukihiro "Matz" Matsumoto in the mid-1990s. Ruby syntax is very readable and often looks like natural, human-readable and -writable pseudo code. 

## Ruby on Rails

### Rails Principles
1. **Don't Repeat Yourself (DRY):** DRY is a principle of software development which states that "Every piece of knowledge must have a single, unambiguous, authoritative representation within a system."
2. **Convention over Configuration (CoC):** Rails has a set of conventions which help you to streamline your application development.

### Rails Components
1. **Action Pack:** This is the component responsible for handling and responding to web requests.
2. **Active Record:** This is the component that provides an interface and binding between the tables in a relational database and the Ruby program code that manipulates database records.
3. **Action Mailer:** This is a framework for designing email service layers.
4. **Active Model:** Provides a defined interface for interacting with Object-Relational Mapping (ORM) implementations.
5. **Action View:** Handles template lookup and rendering.
6. **Active Job:** Provides a framework for declaring jobs and making them run on a variety of queuing backends.
7. **Active Support:** A collection of utility classes and standard Ruby library extensions.

## Getting Started with Rails

### Installation
To install Rails, you need Ruby installed on your system, then you can install Rails via RubyGems.

### Rails Application Structure
A Rails application has

 a standard structure. It includes app, bin, config, db, lib, log, public, test, tmp directories, and several files.

### Rails Command Line Tools
The `rails` command in terminal has several options that can help in creating routes, models, controllers, and more.

## Building your First Rails Application

### Creating a New Rails Application
You can create a new Rails application using the `rails new` command.

### Understanding the MVC Architecture
In Rails, MVC comprises three interconnected parts: Models (Active Record), Views (Action View), and Controllers (Action Controller).

### Creating a New Resource
A resource is the term used for a collection of similar objects, such as articles, people, etc. You can create a new resource using `rails generate scaffold`.

### Running the Application
You can run the application locally using the `rails server` or `rails s` command.

## Rails MVC Structure

### Models
Models in Rails use the Active Record framework, which provides a way to interact with the application's database.

### Views
Views are what the user sees—the HTML, CSS, and optionally JavaScript that make up the user interface.

### Controllers
The controller coordinates the interaction between the user, the views, and the model.

## Database and Active Record

### Configuring a Database
Rails supports several databases (SQLite, PostgreSQL, MySQL). You can configure your database in the `config/database.yml` file.

### Migrations
Migrations allow you to alter your database schema over time in a consistent way.

### Validations
Active Record allows you to validate the state of a model before it gets written into the database.

### Associations
Active Record allows you to declare associations between models (`belongs_to`, `has_one`, `has_many`, `has_many :through`, `has_one :through`, `has_and_belongs_to_many`).

## Conclusion

This guide provides a broad overview of Ruby and Rails, but there are many more details and nuances to learn as you dig deeper. Good luck on your journey into Ruby on Rails!

### References
1. [The Ruby Programming Language](https://www.ruby-lang.org/en/documentation/)
2. [Ruby on Rails Guides](https://guides.rubyonrails.org/)
3. [Ruby on Rails API](https://api.rubyonrails.org/)
4. [Ruby Gems](https://rubygems.org/)
5. [The Rails Command Line](https://guides.rubyonrails.org/command_line.html)
