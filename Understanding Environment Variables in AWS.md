#bookmark 

### Description:

[[AWS]] is a secure cloud services platform that offers a wide array of services. For web applications, services such as **AWS Elastic Beanstalk** and **AWS Lambda** are commonly used. Both these services allow you to set environment variables through the [[AWS Management Console]], CLI, or SDKs. Here we will discuss how to do it using the **AWS Management Console**.

### AWS Elastic Beanstalk

[[AWS Elastic Beanstalk]] is a fully managed service that makes it easy to deploy, run, and scale web applications. Here's how to set environment variables through the Elastic Beanstalk console:

1. Open the [Elastic Beanstalk console](https://console.aws.amazon.com/elasticbeanstalk/).

2. In the **"Regions"** list, select your AWS Region.

3. In the navigation pane, choose **"Environments"**.

4. In the list of environments, choose the name of your environment from the list.

5. In the navigation pane, choose **"Configuration"**.

6. In the **"Software"** configuration category, choose **"Edit"**.

7. Under **"Environment properties"**, specify the environment variables. Each environment property is a key-value pair. For example, to set a `SECRET_KEY` variable, you might enter `SECRET_KEY` for the key and `mySuperSecretKey` for the value.

8. Choose **"Apply"**.

The changes will take effect after your environment is updated. 

### AWS Lambda

[[AWS Lambda]] lets you run your code without provisioning or managing servers. Here's how to set environment variables for a Lambda function:

1. Open the [AWS Lambda console](https://console.aws.amazon.com/lambda/).

2. In the navigation pane, choose "Functions".

3. Choose the name of your function from the list.

4. Under the function's "Configuration" tab, choose "Environment variables".

5. Choose "Edit".

6. Under "Environment variables", add your key-value pairs. For example, to set a `SECRET_KEY` variable, you might enter `SECRET_KEY` for the key and `mySuperSecretKey` for the value.

7. Choose "Save".

The changes will take effect immediately.

### How it Works in Principle 

Environment variables in AWS are **encrypted** and **decrypted** after they leave the service and before they enter the service. However, if you need to store highly sensitive information, consider using AWS Secrets Manager or AWS Systems Manager Parameter Store, which are designed for storing sensitive configuration data.

### Best Practice Example

this example will use beanstalk...

When deploying an `Express.js` application on **AWS Elastic Beanstalk**, the application is expected to run on port `8080`. This port number is provided to your application through an environment variable `process.env.PORT`. 

To set up your Express application to use this environment variable, follow these steps:

1. In your main application file (e.g., `app.js`), when setting up the server to listen, use `process.env.PORT` as your port number.

Example:

```javascript
var express = require('express');
var app = express();

var port = process.env.PORT || 3000;
app.listen(port, function () {
  console.log('App listening on port ' + port);
});
```

In this code, if `process.env.PORT` is defined, your application will use that. If not, it will default to port `3000`. This allows flexibility between your development environment (where `process.env.PORT` might not be defined and you'd want to default to `3000`), and your production environment on Elastic Beanstalk (where `process.env.PORT` will be defined as `8080`).

2. Make sure your project structure includes:

    - Your main application file (like `app.js`)
    - A `Procfile` in the root directory, with the command to start your application (e.g., `web: node app.js`)
    - A `package.json` file in the root directory, which lists your project's npm dependencies.

---

By following these steps, your Express.js application will be able to run on the correct port in the AWS Elastic Beanstalk environment.


### Summary 

Setting environment variables in AWS allows you to manage application configuration securely and separately from your code, enhancing your application's security and scalability (in otherwords, it **decouples** it). Always ensure that sensitive information like API keys, database credentials, or secret keys are never stored directly in your code or version-controlled system.
