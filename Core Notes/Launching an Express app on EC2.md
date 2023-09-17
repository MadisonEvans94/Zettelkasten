#seed 
upstream: [[EC2]]

### Introduction: 

This section will guide you through the process of setting up an **`Express.js`** server on an **AWS EC2** instance.

*Here are the steps to launch an EC2 instance that will be covered in this document:*

1. Sign in to the **AWS Management Console**.
2. Choose **Launch Instance**.
3. In the Choose an **Amazon Machine Image (AMI)** page, choose an AMI.
4. In the **Choose an Instance Type** page, choose the type of instance.
5. Configure the instance.
6. Add storage to the instance.
7. Tag the instance.
8. Configure the security group.
9. Review and launch the instance.



### 1. Sign in to the AWS Management Console

To start, navigate to the [AWS Management Console](https://console.aws.amazon.com/). If you have an AWS account, you can sign in. If you don't, you'll need to create an account.

### 2. Choose Launch Instance

From the **AWS Management Console**:

1. Go to the `Services` dropdown in the top left corner.
2. Under `Compute`, choose `EC2`.
3. In the EC2 dashboard, choose `Launch Instance`.

### Choose an Amazon Machine Image (AMI)

An **AMI** is a template that contains the software configuration (operating system, application server, and applications) required to launch your instance. See [[AMI]] for more

1. You'll see a list of basic configurations. For a `Node.js` application, choose `Amazon Linux 2 AMI (HVM), SSD Volume Type` which supports `Node.js`.

### Choose an Instance Type

Instance types determine the hardware of the host computer. For a basic Express.js app, a `t2.micro` should be sufficient. Remember, you can change this later as your needs evolve.

### Configure the Instance

Configure the instance to suit your needs. Here you can choose the number of instances, network settings, and more. For a simple setup:
1. Leave `Number of instances` at `1`.
2. Leave the `Network` settings as default.
3. For `IAM role`, choose a role that has the necessary permissions, or leave it blank for now.
4. Leave the `Shutdown behavior` as `Stop`.

### Add Storage

Next, add a storage device to your instance. The default should be sufficient for a basic app, but make sure to adjust this based on the requirements of your application.

### Tag the Instance

Tagging allows you to easily identify your instances in the AWS console. You can add a tag with a key of `Name` and a value of your choice.

### Configure the Security Group

Security groups act like a firewall for your instance. You need to set one up that allows SSH and HTTP/HTTPS access.
1. Choose `Create a new security group`.
2. Name it and give it a description.
3. You will need to add two rules: 
    - Type: `SSH`, Protocol: `TCP`, Port Range: `22`, Source: `My IP`. This allows you to SSH into your instance.
    - Type: `HTTP`, Protocol: `TCP`, Port Range: `80`, Source: `Anywhere`. This allows HTTP access to your Express app.

### Review and Launch the Instance

Review the settings for your instance. If everything looks good, click `Launch`.
1. You will be asked to create a new key pair or use an existing one. This is necessary for SSHing into your instance. Choose `Create a new key pair`, name it, and download it. **Do not lose this file as AWS does not keep a copy.**
2. After the key pair is downloaded, choose `Launch Instances`.

### Deploying Express.js app on EC2

To deploy your Express.js application on EC2:
1. SSH into your EC2 instance using the key pair you downloaded.
2. Clone your Express.js application repository.
3. Install Node.js and npm.
4. Install your application dependencies using `npm install`.
5. Start your application using `node app.js` or `npm start`.

Congratulations, you have launched an Express.js application on an AWS EC2 instance!
