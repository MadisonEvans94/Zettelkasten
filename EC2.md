#evergreen1 
upstream: [[AWS]]

---

**video links**: 

---

## Common EC2 Use Cases

EC2 is a very versatile service and it can be used in a variety of applications. Here are some of the most common use cases for EC2 within common architectures:

### 1. Hosting Web Applications

EC2 is commonly used to host web applications, including websites and web APIs. You can run your web server on an EC2 instance and store your data in an EBS volume or RDS database. 

### 2. Data Processing

You can use EC2 instances to process data. For instance, if you have a large dataset that needs to be processed, you can use EC2 instances to process the data in parallel. This is commonly done in big data and scientific computing applications.

### 3. Batch Processing

Batch processing is a method of running high-volume, repetitive tasks without user interaction. EC2 instances can be used to process these tasks in the background and then be shut down when they're no longer needed, which can save on costs.

### 4. Gaming Servers

EC2 instances can be used to host multiplayer gaming servers. For games that require a lot of compute power, you can use high-performance EC2 instances.

### 5. Content Delivery and Media Transcoding

EC2 instances can be used for content delivery and media transcoding. You can use EC2 instances to convert media files from their source format into versions that will play on devices like smartphones, tablets, and PCs.

### 6. Machine Learning and AI

EC2 provides instances that are optimized for machine learning (ML) and artificial intelligence (AI) workloads. These instances provide access to NVIDIA GPUs for high-performance computing tasks.

### 7. Backup and Recovery

EC2 instances can be used for backup and recovery solutions. For instance, you could take regular snapshots of your EBS volumes and store them in S3 for long-term storage.

Remember, these are just some of the ways you can use EC2. Its flexibility and variety of instance types make it suitable for many different applications.
## EC2 Instance Types Basics
Amazon EC2 provides a variety of instance types optimized to fit different use cases.
 
![[Screen Shot 2023-06-30 at 3.04.43 PM.png]]

### Instance Types
#### General Purpose Instances: 
These instances provide a balance of compute, memory, and networking resources, and are a good choice for many applications.

#### Compute Optimized Instances: 
These instances are ideal for compute-bound applications that benefit from high-performance processors.

#### Memory Optimized Instances: 
These instances are designed to deliver fast performance for workloads that process large data sets in memory.

#### Storage Optimized Instances: 
These instances are designed for workloads that require high, sequential read and write access to very large data sets on local storage.

#### Accelerated Computing Instances: 
These instances use hardware accelerators, or co-processors, to perform functions such as floating-point number calculations, graphics processing, or data pattern matching more efficiently than software on a general-purpose CPU.

### Naming Convention 
The instance name is generally made up of characters and numbers that can be broken down into **3** parts. Let's use **`m5.large`** as an example 

#### The first character(s) indicates the instance **family**
This generally describes the **balance of resources** for the instance. For example, `m` stands for "general purpose", `t` stands for "[[Burst-able Performance]]", `c` stands for "compute optimized", `r` stands for "memory optimized", `i` stands for "storage optimized", `p` stands for "GPU instance", `g` stands for "Graphics optimized", `f` stands for "FPGA instance", and `x` stands for "Extreme Memory Optimized".

#### The number following the instance family indicates the instance **generation**
AWS increases this number whenever it updates the hardware of the instance family. For instance, `m5` would be a newer generation than `m4`.

#### The part after the dot indicates the instance **size** 
The part after the dot indicates the instance size within the specific family and generation. For example, in `m5.large`, `large` indicates the size of the instance. The names here follow a logical progression from `nano` and `micro` to `small`, `medium`, `large`, `xlarge`, and then multiples of `xlarge` like `2xlarge`, `4xlarge`, etc. The larger the size, the more CPU, memory, storage, and network capacity it has.

---


## EC2 User Data

**EC2 User Data** is a feature that allows you to include some custom scripts or metadata when launching your EC2 instance.

Follow these steps to create an EC2 instance with User Data:

1. From the AWS Console, navigate to the EC2 Dashboard and click `Launch Instance`.

2. In the 'Choose an Amazon Machine Image (AMI)' step, select an AMI. For a basic website, the **Amazon Linux 2 AMI** is a good choice.

3. In the 'Choose an Instance Type' step, select a suitable instance type (such as t2.micro, which is in the AWS free tier).

4. In the 'Configure Instance' step, navigate to the `Advanced Details` section. Here, you can enter your User Data script. 

    A basic User Data script to start a web server might look like:

    ```
    #!/bin/bash
    yum update -y
    yum install -y httpd
    systemctl start httpd
    systemctl enable httpd
    echo "Hello from EC2 instance" > /var/www/html/index.html
    ```

5. Continue with the launch process (including adding storage, tags, and configuring the security group).

6. Review the settings and click `Launch`.

## Most Common Bugs

While launching and managing EC2 instances, developers may encounter a number of issues. Here are some of the most common mistakes to watch out for:

### 1. Incorrect Security Group Configuration

One of the most common issues is incorrectly configuring security groups. Be sure to open the necessary ports for your applications, such as port 22 for SSH, port 80 for HTTP, and port 443 for HTTPS. But, remember not to open all the ports which may lead to potential security risks.

### 2. Not Assigning Correct IAM Role

Another common issue is forgetting to assign the correct IAM role to an EC2 instance. This can cause issues if your application needs to access other AWS services. Always ensure that you have the correct IAM roles and policies in place before launching your instance.

### 3. Incorrect Instance Type/Size Selection

Choosing the wrong instance type or size is another common mistake. If you select a type or size that's too small, your application may run out of resources and perform poorly. On the other hand, if it's too large, you may end up paying more than necessary.

### 4. Incorrect AMI

Using the wrong AMI (Amazon Machine Image) for your application can cause problems. Be sure to use an AMI that includes the correct operating system, software, and configuration settings required by your application.

### 5. Launching in the Wrong Availability Zone

If you launch your EC2 instance in the wrong Availability Zone, you may encounter latency issues or additional data transfer costs, especially if your other resources (like RDS instances or EFS file systems) are in a different zone. 

### 6. EBS Volume Errors

Errors can occur if an EBS volume is not correctly attached or mounted to your EC2 instance. Make sure the device names match when attaching a volume, and that the file system on the volume is correctly mounted.

### 7. Running Out of EBS Space

Another common problem is running out of EBS space. Monitor your disk usage and if necessary, increase the size of your EBS volumes or free up space on your volumes.

### 8. Not Using Key Pairs Correctly

Mistakes with key pairs, such as losing the private key or not having the correct permissions on your key pair file can lock you out of your instances.

To avoid these and other common mistakes:

- Thoroughly understand EC2 concepts and components
- Review all settings before launching an instance
- Regularly monitor and maintain your instances
- Follow AWS best practices and recommendations.
## Security Groups and Classic Ports Overview

**Security Groups** in EC2 act as a virtual firewall that controls the traffic for one or more instances. 

> For more info on security groups, see [[AWS Security Groups]]

Each security group consists of a set of rules, which filter traffic by allowing it to reach the various instances that are associated with the security group. The rules of a security group can be modified at any time, with the new rules being automatically applied to all instances associated with the security group.

Common ports to keep in mind:

- Port **22**: SSH (Secure Shell)
- Port **80**: HTTP (Hypertext Transfer Protocol)
- Port **443**: HTTPS (HTTP Secure)

---
## SSH Overview

**SSH**, or Secure Shell, is a protocol that you can use to securely log onto remote systems in your EC2 instances. With an SSH client, you can connect to your EC2 instance and run commands as if you were sitting in front of the machine. The command line code for connecting to your EC2 instance is as follows: 

```bash 
ssh -i /path/my-key-pair.pem ec2-user@public-ip-of-your-instance
```

> Note: you need to generate a `.pem` or `.cer` file with key credentials in order to execute the command above. See [[Connecting to EC2 via ssh]] for more


SSH connections are encrypted and secure, involving a pair of keys: a private key that stays with the client, and a public key that is placed on the server. Only the client with the private key can initiate a secured session with the server.

---

## EC2 Instance Roles

**Instance Roles** are permissions that you can grant to your application to make AWS API requests. When you launch an instance, you can specify an IAM role that applications running on the instance can use to access AWS resources.

You use IAM roles to delegate access to your AWS resources. With IAM roles, you can establish trust relationships between your trusting account and other AWS trusted accounts.

Here are examples of IAM role permissions that could be created for EC2 instances:

1. **S3FullAccess**: An IAM role with full access to S3. The applications on the EC2 instance can create, read, update and delete S3 objects.
    
2. **DynamoDBFullAccess**: An IAM role with full access to DynamoDB. The applications on the EC2 instance can perform all actions on DynamoDB tables.
    
3. **RDSReadOnlyAccess**: An IAM role with read-only access to RDS. The applications on the EC2 instance can read data from RDS databases but cannot modify them.
    
4. **EC2DescribeInstances**: An IAM role that allows applications on the EC2 instance to describe EC2 instances.
    
5. **LambdaExecute**: An IAM role that allows applications on the EC2 instance to invoke AWS Lambda functions.

---
## EC2 Instance Purchasing Options

Amazon EC2 provides several purchasing options to enable you to optimize your costs. They include:

### On-Demand Instances: 
Pay for compute capacity by per hour or per second depending on the instances that you launch.

#### Example Scenario: 
On-Demand instances are useful for short-term, irregular workloads that cannot be interrupted. They are best when you have a sudden increase in traffic, like during a product launch event.

### Reserved Instances:
Provide you with a significant discount (up to 75%) compared to On-Demand instance pricing and provide a capacity reservation when used in a specific Availability Zone. **1** or **3** year rental option

#### Example Scenario: 
Reserved instances are ideal for predictable workloads with steady-state usage. For example, if you are running a database server that needs to be up and running 24/7, a reserved instance would be a cost-effective choice.

### Spot Instances: 
Allow you to request spare Amazon EC2 computing capacity for up to 90% off the On-Demand price.

#### Example Scenario: 
Spot Instances are useful for workloads with flexible start and end times, or that can withstand interruptions. For instance, you could use Spot Instances for big data and analytics, image and media rendering, testing, and other **non-time sensitive** tasks.

### Dedicated Hosts: 
Physical EC2 server dedicated for your use, which can help you reduce costs by allowing you to use your existing server-bound software licenses.

#### Example Scenario: 
Dedicated hosts are often used for regulatory requirements that may not support multi-tenant virtualization, or for workloads where licensing costs (such as for Windows Server or SQL Server) can be significantly reduced by using dedicated physical servers.


## EC2 Instance Storage 
see [[EC2 Instance Storage]] for more details 