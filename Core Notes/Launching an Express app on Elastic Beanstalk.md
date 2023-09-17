#incubator 

upstream: [[AWS Elastic Beanstalk]]

[Deploy a Node.js application on AWS Elastic Beanstalk](https://www.youtube.com/watch?v=dWfng4qJYMU&ab_channel=DigitalCloudTraining)

## Introduction: 

This section will guide you through the process of setting up an **`Express.js`** server on **AWS Elastic Beanstalk** 

*Here are the steps to deploy an Express.js application using Elastic Beanstalk that will be covered in this document:*
1. Prepare your `Express.js` application for the AWS environment.
2. Create an Elastic Beanstalk environment.
3. Upload your application.
4. Monitor your application.

## Workflow: 

### 1. Prepare your `Express.js` application for the AWS environment

Before you can deploy your `Express.js` application to AWS Elastic Beanstalk, you need to add a `Procfile` to your project root directory. This file is used by AWS to start your application.

Create a `Procfile` with the following line:

```
web: node app.js
```

Replace `app.js` with the entry point to your application if it's different.

Your application will run on port `8080` in the AWS environment, so make sure your application is set to listen on `process.env.PORT`. see [[Understanding Environment Variables in AWS]] for more details 

```javascript
const port = process.env.PORT || 3000;
app.listen(port);
```

### 2. Create an Elastic Beanstalk environment

#### 1. Open the [Elastic Beanstalk console](https://console.aws.amazon.com/elasticbeanstalk): 
#### 2. Click `Create a new environment`: 
#### 3. Choose `Web server environment` and click `Select`.

![[Screen Shot 2023-06-22 at 11.45.54 AM.png]]

#### 4. In the `Environment information` section, specify the following:
- **Application**: If you have not created an application yet, enter a new application name.
- **Environment name**: Enter a name for your environment.
- **Domain**: Choose a unique DNS name.
#### 5. In the `Platform` section, choose `Node.js`: 

*for other runtime environments such as `Python` or `Ruby`, you would simply select those during this step*

![[Screen Shot 2023-06-22 at 11.49.48 AM.png]]
#### 6. Click `Create environment`: 

### 3. Zipping your application for upload

*While your environment is being created, you can prepare your application version for upload:*

You can use the `zip` command-line utility to create a source bundle for your Elastic Beanstalk application

![[Screen Shot 2023-06-23 at 10.12.21 AM.png]]

#### Here's a basic command that will create a zip file of your project:

```bash
zip -r my-application.zip .
```

This command will recursively (`-r`) zip all files in your current directory (`.`) and put them into a file called `my-application.zip`.

However, when creating a zip file for an Elastic Beanstalk application, **you want to exclude certain files and directories**, like the `node_modules` directory and any files that are defined in your `.gitignore` or `.ebignore` file. 

#### To exclude the `node_modules` directory: 
...you can modify the command like so:

```bash
zip -r my-application.zip . -x "node_modules/*"
```

This command will zip all files in your current directory except for those in the `node_modules` directory.

#### To exclude files that are defined in a `.gitignore` or `.ebignore` file
...you can use the `zip` command with `git`:

```bash
git archive -v -o my-application.zip --format zip HEAD
```

This command uses `git` to archive your project (excluding files listed in your `.gitignore`), and outputs (`-o`) the archive to a zip file called `my-application.zip`. Note that this command only includes tracked files in git. If there are necessary files for your application that are not tracked by git, they won't be included in the zip file.

**Remember to run these commands in the root directory of your project** where your `package.json` file is located. Elastic Beanstalk expects your `package.json` to be in the root of your zipped file.

After you've created your zip file, you can upload it to Elastic Beanstalk to deploy your application.

1. In the AWS Elastic Beanstalk console, under `Running versions`, choose `Upload and Deploy`.
2. Choose `Choose file` and select the `.zip` file you created.
3. Enter a version label, or use the automatically generated one.
4. Click `Deploy`.

*This will upload your application to Elastic Beanstalk and deploy it to your environment.*

### 4. Configure service access

![[Screen Shot 2023-06-22 at 3.17.47 PM.png]]

#### Service Role: 

The service role is an [[IAM]] role that AWS Elastic Beanstalk assumes to use AWS resources on your behalf. This role provides permissions that determine what other AWS service resources your Beanstalk app can access. 

This role is necessary because it ensures that your Elastic Beanstalk environment has the right permissions to access resources that it needs to operate, such as **reading from an S3 bucket**, **writing logs to [[Amazon CloudWatch]]**, **accessing a database**, and more.

When creating a new Elastic Beanstalk environment, you can either choose an existing service role or create a new one. If you're unsure what to do, *I would recommend letting Elastic Beanstalk create a new service role for you*, as it will automatically assign the necessary permissions.

#### EC2 Key Pair:

An [[EC2]] key pair consists of a **private key** that you keep safe and a **public key** that you upload to AWS. Together, they allow you to connect to your instances securely. When you launch an instance, you specify the name of the key pair. When you connect to your instance, you provide your private key instead of a password.

The dropdown list for "Select an EC2 key pair" in the AWS Elastic Beanstalk interface should show all the EC2 Key Pairs that are currently available in your AWS account and in the current region. If the dropdown list is empty, it means you haven't yet created any EC2 Key Pairs in your account in this region. 

**If you want to be able to SSH into your instances, you'll need to create a key pair. Here's how to do it:**

1. Open the Amazon EC2 console at https://console.aws.amazon.com/ec2/.
2. In the navigation pane, under "Network & Security", choose "Key Pairs".
3. Choose "Create key pair".
4. For "Name", enter a descriptive name for the key pair. For example, you might name your key pair "MyBeanstalkApp".
5. For "File format", choose the format in which to save the private key. To save the private key in a format that can be used with OpenSSH or ssh on Linux, choose "pem". To save the private key in a format that can be used with PuTTY on Windows, choose "ppk".
6. Choose "Create key pair".
7. To download the private key file (.pem or .ppk), choose "Download". Store the private key file in a safe place.

Your new key pair should now appear in the dropdown list when you refresh the Elastic Beanstalk "**Create new Environment**" page.

Remember to keep your private key file secure and do not share it. Anyone who has access to your private key can connect to your instances. Also, you can't download the private key file again after it's created.

*To put it simply*, you use the EC2 key pair for [[SSH]] access to your Elastic Beanstalk EC2 instances if you need to directly access the server for debugging or other reasons. If you don't foresee needing SSH access, you don't have to create or use a key pair. However, it can be useful for troubleshooting purposes.

#### EC2 Instance Profile:

An **EC2 instance profile** is an IAM role that you can attach to your instance. This role provides permissions that determine what other AWS service resources the instance can access. 

In the context of Elastic Beanstalk, the instance profile is used to specify what permissions the EC2 instances in your environment have. Similar to the service role, the instance profile is important for giving your application the necessary permissions to access other AWS services.

*Remember*, good security practice dictates that you provide the least amount of privilege necessary for your application to function. It can be easy to give too many permissions in the interest of 'making things work', but this can lead to potential security vulnerabilities.

#### TLDR: 

- **Service Role**: Permissions for the Elastic Beanstalk service.
- **EC2 Key Pair**: Allows secure SSH access to your instances.
- **EC2 Instance Profile**: Permissions for the instances running your application.

### 5. Set up networking, database, and tags (optional)

... info goes here ...

### 6. Configure instance traffic and scaling

... info goes here ...

### 7. Configure updates, monitoring, and logging

... info goes here ...

### Once you have successfully set up your application... 

you will be able to access it through a URL that Elastic Beanstalk generates for you. This URL is often referred to as the Elastic Beanstalk environment URL.

If you want a more user-friendly URL (like `www.myapp.com`), you would need to register a domain and then configure DNS settings to point your domain to the Elastic Beanstalk environment URL. This can be done through services like [[AWS Route 53]] or any other domain registration/DNS service.

## Monitoring 

AWS Elastic Beanstalk provides several monitoring tools:

- **Dashboard**: Gives you a high-level overview of the environment.
- **Health**: Provides detailed information about the resources running in the environment.
- **Events**: Shows a history of the changes that have occurred in the environment.
- **Logs**: Allows you to view logs.
- **Monitoring**: Shows various metrics of your environment.

*Make sure to regularly check these tools*, especially when deploying a new version of your application or changing the environment configuration. See [[An in-depth look at elastic beanstalk monitoring]] for more details 


## Conclusion

**AWS Elastic Beanstalk** provides an easy-to-use service for deploying and scaling web applications and services. By simply uploading your code, you can achieve application deployment, capacity provisioning, load balancing, auto-scaling, and application health monitoring.