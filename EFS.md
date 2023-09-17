#incubator 
upstream:

---

**video links**: 

---

# Amazon EFS (Elastic File System)

Amazon Elastic File System (EFS) is a scalable and managed elastic NFS file system for use with AWS Cloud services and on-premises resources.

## What is EFS?

EFS is a file storage service for use with Amazon EC2. It allows you to create and configure file systems quickly and easily. EFS is built to elastically scale on demand without disrupting applications, growing and shrinking automatically as you add and remove files. It enables your applications to have a shared access to a file system and is accessible by multiple EC2 instances from multiple Availability Zones.

## Why Use EFS?

EFS is particularly useful for applications and workloads that require shared access to files or that require scale-out and high-availability storage. Some common use cases include content management, web serving, data analytics, media processing workflows, container storage, and software development and testing.

## How Does EFS Work?

Under the hood, EFS works by distributing data across multiple Availability Zones in an AWS region. It ensures that your data is reliable, robust, and durable. The file system you create supports concurrent read and write access from multiple EC2 instances and it provides consistent low latencies.

Each EFS file system object (i.e., directory, file, and link) is redundantly stored across multiple Availability Zones. Unlike traditional file servers, EFS doesn't require you to provision storage in advance. It automatically adds and removes storage as you add and remove files.

## EFS Access Points

EFS Access Points are application-specific entry points into an EFS file system that allow you to manage application access to shared datasets. Access Points can enforce a user identity, including the user's POSIX groups, for all file system requests made through the Access Point. They can also enforce a different root directory for the file system so that clients can only access data in that directory or its subdirectories.

Access Points can be created for specific applications or sets of applications. Each Access Point has properties for the POSIX user and group, and root directory. When an application uses an Access Point to access an EFS file system, the application’s POSIX user and group override any identity provided by the NFS client. The root directory specified in the Access Point is exposed as the root of the file system to the client connecting through the Access Point.

## How to Set Up and Use EFS?

Setting up and using EFS involves the following steps:

### 1. Create a File System

To create a file system, navigate to the EFS console on AWS, and choose "Create file system". Follow the prompts and customize the settings as needed. Some settings include enabling encryption, choosing performance mode (General Purpose or Max I/O), and choosing throughput mode (Bursting or Provisioned).

### 2. Configure Network Access

Next, you'll need to configure the network access. Here, you specify which VPC the EFS file system will be accessible from, and create mount targets in your subnets. A mount target is essentially a network interface that EC2 instances can use to connect to the EFS file system.

### 3. Configure Security

Configure the security groups. These act like a firewall, and control the traffic allowed to reach the mount targets. Typically, you'll want to allow inbound access to TCP port 2049, which is used by NFS.

### 4. Create and Configure Access Points

After the file system and network access are set up, you can create Access Points from the EFS console. Here, you can configure the POSIX user, POSIX group, and root directory for the Access Point.

### 5. Mount the File System

After the EFS file system is created, configured, and Access Points are set up, you can mount it

 on your EC2 instances with the `mount` command. You'll need the `amazon-efs-utils` package installed on your EC2 instances. 

### 6. Use the File System

Once the EFS file system is mounted, you can use it like any other file system. You can navigate into it with `cd`, create directories with `mkdir`, create files with `touch`, and so on. All EC2 instances mounting the EFS file system can read and write to the same files.

For more details, always refer to the [AWS EFS documentation](https://aws.amazon.com/efs/).


