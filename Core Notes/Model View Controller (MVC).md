#incubator 
###### upstream: [[Software Development]]

### Overview: 

The Model-View-Controller (MVC) pattern is a design pattern often used in web applications to structure code in a logical and intuitive way. The pattern divides application logic into three interconnected components:

-   **Model**: The model represents the data and the rules that govern access to and updates of this data. In other words, all your database interactions and data-related logic (like validations and computations) would go here.
    
-   **View**: The view is what the user sees and interacts with (UI). It displays the model's data for the user and may also provide interfaces for user interaction. In the context of an Express.js application, these would be your templates rendered on the client-side.
    
-   **Controller**: The controller receives user input and makes calls to model objects and views to perform appropriate actions. All your routes and business logic would be here.

### Analogy: 

Okay, let's think about the MVC pattern in terms of making a pizza at a pizza place!

1.  **Model (The Kitchen)**: This is where the ingredients (data) are stored and the pizza (data processing) is made. All the magic and secret recipes (business logic) are here. The kitchen (Model) has all the information about how to prepare the pizza, like how much cheese to use, how long to bake it, etc.
    
2.  **View (The Dining Area)**: This is where you see and eat the pizza. It's the beautiful part that customers interact with. In a web app, it's what you see on the screen - the layout, colors, fonts, etc.
    
3.  **Controller (The Waiter)**: The waiter (Controller) takes your order (input), goes to the kitchen (Model) to make your pizza (processes data), and then brings the pizza to your table (View) for you to enjoy. If you want to modify your order (like adding more cheese or removing an ingredient), the waiter communicates this to the kitchen too.
    

So, in summary, the Controller takes your requests and tells the Model. The Model processes the request and sends it back to the Controller. Finally, the Controller presents it to you via the View.

Remember, this is a simplified analogy. The real MVC pattern has more details, especially when used in different programming languages or frameworks, but this gives you a basic understanding.

### Putting it All Together: 

In a full stack express application with React as the front end, the MVC architecture is as follows: The static js/html/css bundle that is served to the client is the view. The ORM schema set up in the express app is the model. And the route handlers you set up in your Express app is the controller. These route handlers receive HTTP requests from the client, interact with the Model (using Sequelize, Mongoose, or another ORM) to retrieve or manipulate data as necessary, and then send a response back to the client.

*In summary, in a full-stack Express and React application:*

-   The **View** is the React app.
-   The **Model** is defined by your ORM schemas.
-   The **Controller** is the route handlers in your Express app.


### Example: 

*Here's an example of how you might structure an Express.js application using the MVC pattern. This is a simplified example of a blog system:*

1. **Model** (in `models/Post.js`):
```js
import mongoose from 'mongoose';

const PostSchema = new mongoose.Schema({
  title: String,
  body: String,
});

export default mongoose.model('Post', PostSchema);
```
*see [[mongoose]] for more details*

if you are using a relational database, then you should use an ORM like **Sequelize**. This code would fall under the category of model in the MVC pattern. 

2. **View** (in `views/post.ejs`):
```html
<h1><%= post.title %></h1>
<p><%= post.body %></p>
```
*If you're using React (or any other frontend JavaScript library/framework like Vue or Angular) for your frontend, the concept of "view" in MVC shifts from the server-side to the client-side.*

In a traditional Express.js application, the views are usually server-side rendered HTML pages (using a templating engine like EJS, Pug, Handlebars, etc.). However, when you're using React, the views are essentially your React components. React takes care of rendering your components into HTML and updating them when your state changes.

The Express.js backend becomes a RESTful API or GraphQL API server that your React application communicates with. The server is responsible for handling requests, interacting with the database, and sending responses back to the client. It doesn't concern itself with views anymore.

Here's how it might look:

**View** (now a React component, `src/Post.js`):
```js
import React from 'react';

function Post({ post }) {
  return (
    <div>
      <h1>{post.title}</h1>
      <p>{post.body}</p>
    </div>
  );
}

export default Post;
```

3. **Controller** (in `controllers/postController.js`):
```js
import Post from '../models/Post.js';

const postController = {
  getAll: async (req, res) => {
    const posts = await Post.find();
    res.render('posts', { posts: posts });
  },

  getOne: async (req, res) => {
    const post = await Post.findById(req.params.id);
    res.render('post', { post: post });
  },

  // more methods for creating, updating, deleting posts...
};

export default postController;
```

4. Finally, you would wire up your controllers in your routes (in `routes/posts.js`):
```js
import express from 'express';
import postController from '../controllers/postController.js';

const router = express.Router();

router.get('/', postController.getAll);
router.get('/:id', postController.getOne);

// more routes for creating, updating, deleting posts...

export default router;
```

### Summary: 

*In the MVC architecture, the "Model" handles data and business logic, the "View" presents and captures user interaction, and the "Controller" processes user requests and updates the Model and View accordingly.*