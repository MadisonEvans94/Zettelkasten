#seed 
upstream:

---

**links**: 

---

Brain Dump: 

--- 



Django includes several built-in applications that provide common web development functionalities. Here's an overview of each of the default apps included in the `INSTALLED_APPS` setting of a new Django project:

1. **django.contrib.admin**
   - **Purpose**: Provides an admin interface for your Django project. This is a powerful feature that allows you to manage the data in your application's models.
   - **Common Use**: Used to quickly create an interface for CRUD (Create, Read, Update, Delete) operations on your models.

2. **django.contrib.auth**
   - **Purpose**: Handles authentication and authorization.
   - **Common Use**: Provides the User model and associated functionalities like login, logout, and password management. It also handles permissions and groups.

3. **django.contrib.contenttypes**
   - **Purpose**: Allows Django to track all of the models installed in your application, providing a high-level, generic interface for working with your models.
   - **Common Use**: Used by other Django components such as the authentication app to link models with permissions. It facilitates generic relations between models.

4. **django.contrib.sessions**
   - **Purpose**: Manages sessions in your Django application.
   - **Common Use**: Handles the storage of arbitrary data that you want to persist between requests, typically using a cookie that contains a session ID.

5. **django.contrib.messages**
   - **Purpose**: Provides a temporary messaging system for one-time notifications.
   - **Common Use**: Used to display messages like success alerts, error messages, or other user feedback. Messages are stored in one request and retrieved for display in a subsequent request.

6. **django.contrib.staticfiles**
   - **Purpose**: Manages the serving of static files, such as CSS, JavaScript, and images.
   - **Common Use**: Collects static files from each of your applications (and any other places you specify) into a single location that can easily be served in production. 

These apps are included by default because they provide core functionalities that are essential to most web development projects. They are designed to work together but can also be replaced or extended if you need to customize their behavior.
