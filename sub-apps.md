#seed 
###### upstream: 

### Core Thought: 

A **sub-app** in Express.js is essentially an Express instance which is used in another Express instance, similar to [[Middleware]]. This might be used for modularity, as it allows you to separate your app into parts each with its own responsibilities, or it might be used for mounting an app on a specific path, or both.

The `app.mountpath` property in Express.js is used in a sub-app (an Express application) to identify the path where it is mounted on the parent app. It gives us the mounted URL directory, or directories, if your sub-app is mounted on multiple paths.

*TLDR: sub-app = middleware*

#### Example

```js
var express = require('express')

var app = express() // parent app
var admin = express() // sub app

admin.get('/', function (req, res) {
  console.log(admin.mountpath); // /admin
  res.send('Admin Homepage');
})

app.use('/admin', admin); // mounts the sub app
app.listen(3000);
```

In this example, `admin` is a sub-app which is mounted on the parent app `app` at the `'/admin'` path. Inside the `admin` app, you can access `admin.mountpath` to get the path at which it was mounted on the parent app, which in this case will be `'/admin'`.

Now if we navigate to '[http://localhost:3000/admin](http://localhost:3000/admin)', it will bring up the message `'Admin Homepage'` as we have defined in the sub-app.