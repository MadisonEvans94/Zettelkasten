#incubator 
upstream: [[EC2]]

---

**video links**: 

---


## AMI
### Overview 
**Amazon Machine Images (AMIs)** serve as the templates for launching EC2 instances. Each AMI includes a specific operating system, server, and additional software required to launch an instance. AMIs are available in a variety of configurations for different types of instances. AMIs can be either public or private.

### Difference Between AMI and User Data

**Amazon Machine Image (AMI)** and **EC2 User Data** are both important elements in the Amazon Web Services (AWS) ecosystem, especially when it comes to launching and configuring Amazon EC2 instances. However, they serve distinct roles and purposes.

#### AMI (Amazon Machine Image)

An AMI is essentially a template that contains a software configuration (which includes an operating system, application server, and applications). It is used to launch EC2 instances. Once an instance is launched from an AMI, the instance behaves exactly like the image specified. For instance, if the AMI contains the software to be an FTP server, instances launched from this AMI would behave as FTP servers.

#### EC2 User Data

EC2 User Data, on the other hand, is a mechanism for supplying **instance-specific** configuration data when launching your EC2 instances. It is primarily used for running scripts during the instance startup sequence (also known as [[bootstrapping]]). User Data is a way to customize instances beyond what is specified in the AMI.

For example, you might launch all of your instances from the same AMI but use User Data to customize each instance to perform different tasks. One instance might be configured as a web server, another as a database server, and so on.

>In summary, while AMIs are used as a starting point to launch your EC2 instances, EC2 User Data is used to customize these instances during the launch process.

### EC2 Image Builder 
EC2 Image Builder simplifies the creation, maintenance, validation, sharing, and deployment of Linux or Windows Server images. With EC2 Image Builder, you define an image recipe containing your configurations, tests, and updates, which the service uses to produce a reliable and up-to-date machine image.

## EBS
### Overview
**Amazon Elastic Block Store (EBS)** provides raw block-level storage that can be attached to Amazon EC2 instances. These block storage volumes can persist independently from the life of an instance. EBS volumes are highly available, highly reliable volumes that can be leveraged as an EC2 instance’s boot partition or attached to a running EC2 instance as a standard block device. 

### EBS Snapshots  
EBS Snapshots are point-in-time copies of data stored on your EBS volumes. Snapshots can be used as the starting point for new EBS volumes, and to protect data for long-term durability. They can be used for backups, disaster recovery, migration, and other scenarios. 

### EBS Multi-Attach
EBS Multi-Attach allows you to attach a single Provisioned IOPS SSD (io1) volume to up to sixteen Nitro-based EC2 instances that are in the same Availability Zone. Each attached instance has full read and write permission to the shared volume.

### Use Cases 
EBS is often used for databases or other applications that require **frequent read and write** operations, like logging applications or content management systems. 

## EFS
### Overview
**Amazon Elastic File System (EFS)** is a scalable file storage for use with Amazon EC2. You can create an EFS file system and configure your instances to mount the file system. You can use an EFS file system as a common data source for workloads and applications running on multiple instances.

### Use Cases 
EFS is commonly used for content repositories, development environments, web serving, and CMS, and most importantly for shared storage use cases, such as with containerized applications.

## Difference Between EFS, EBS, and S3
**EBS** is block storage that works well for use cases such as databases and raw file storage. 

**EFS** is a shared, elastic file system designed to be used with EC2 instances. It's well-suited for use cases that need a file system structure, like content management systems. 

**S3**, or Simple Storage Service, is an object storage service ideal for storing large, unstructured data sets and for use with data backup, archival, and analytics.
