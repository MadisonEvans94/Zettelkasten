#seed 
upstream: [[Model View Controller (MVC)]]

---

**video links**: 

---

# Brain Dump: 


--- 

## Understanding MVC Architecture with a Web Example <a name="mvc-architecture-example"></a>

The Model-View-Controller (MVC) design pattern is a way to organize code in a way that separates concerns, and is commonly used in web applications.

### Model <a name="model"></a>

The Model represents the data and the rules that govern access to and updates of this data. In web applications, this often means the model interacts with a database. 

In our blog site example, the model would be responsible for managing the data in blog posts. This could include tasks like **retrieving all blog posts**, **retrieving a single blog post**, **creating a new blog post**, **updating a blog post**, **or deleting a blog post**. (so basically our CRUD operations)

In Laravel, each database table has a corresponding "**Model**" that is used to interact with that table. For example, you might have a `Post` model that corresponds to the `posts` table in your MySQL database.

---

### View <a name="view"></a>

The View is the **user interface** — what you see in the browser when you render the web page. It's responsible for displaying the data provided to it by the Controller in the format that the user sees (HTML, CSS, JavaScript).

In our blog site example, the view would be responsible for displaying the blog posts. This could include a page that shows a list of all blog posts, a page that shows a single blog post, or a form for creating or updating a blog post.

>In a Next.js application, the views would be your React components.

---

### Controller <a name="controller"></a>

The Controller handles **user input and interactions**. Upon an event (like a user clicking a button), the controller invokes actions to be performed on the data model or asks the view to update itself.

In our blog site example, the controller would be responsible for handling user actions like viewing a blog post, creating a new blog post, or updating a blog post. 

In Laravel, you would have a `PostController` that handles these actions. For example, when a user wants to view a blog post, the `PostController` would use the `Post` model to retrieve the data for the blog post, and then pass this data to the view.

---

In summary... 

Database Models + CRUD functions = **Model** 
Frontend UI = **View**
User Events = **Controller** 