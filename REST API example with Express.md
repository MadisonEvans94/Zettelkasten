#evergreen1 
upstream: [[REST API]]

---

**video links**: 

---


### Example 
*Let's say you are creating a [[REST API]] API for a simple blog. You might need routes for getting all posts, getting a single post, creating a post, updating a post, and deleting a post. Here's a very basic example of how you might set that up using ES6 syntax with `Express.js`:*


```javascript

const app = express();

app.use(bodyParser.json()); // for parsing application/json

let posts = [
  { id: 1, title: 'First Post', content: 'This is the first post.' },
  { id: 2, title: 'Second Post', content: 'This is the second post.' },
];

// Get all posts
app.get('/posts', (req, res) => {
  res.json(posts);
});

// Get a single post
app.get('/posts/:id', (req, res) => {
  const postId = Number(req.params.id);
  const foundPost = posts.find((post) => post.id === postId);

  if (!foundPost) {
    res.status(404).send({ error: 'Post not found' });
  } else {
    res.json(foundPost);
  }
});

// Create a post
app.post('/posts', (req, res) => {
  const newPost = req.body;
  posts.push(newPost);
  res.status(201).json(newPost);
});

// Update a post
app.put('/posts/:id', (req, res) => {
  const postId = Number(req.params.id);
  const body = req.body;
  const post = posts.find((post) => post.id === postId);
  const index = posts.indexOf(post);

  if (!post) {
    res.status(404).send({ error: 'Post not found' });
  } else {
    const updatedPost = { ...post, ...body };
    posts[index] = updatedPost;
    res.json(updatedPost);
  }
});

// Delete a post
app.delete('/posts/:id', (req, res) => {
  const postId = Number(req.params.id);
  const newPostsArray = posts.filter((post) => post.id !== postId);

  if (!newPostsArray.length) {
    return res.status(404).send({ error: 'Post not found' });
  }
  
  posts = newPostsArray;
  res.status(200).json({ message: 'Post deleted successfully.' });
});

const port = 3000;
app.listen(port, () => console.log(`Server is running on port ${port}`));

```
*see [[what is req.params?]] for some context on the `req.params` object...* 

Please note that this example is simplistic and isn't meant for production use. For example, the posts are stored in memory, so they'll be lost when the server is restarted. In a real-world application, you'd likely use a database for persisting your data. Also, for better error handling and validation, consider using a library like `express-validator` or `joi`. Lastly, remember to consider security measures, such as adding authentication and rate limiting. See [[Security Best Practices in Express]] for more