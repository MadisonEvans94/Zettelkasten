#seed 
upstream: [[AWS]], [[EC2]], [[Computer Engineering]]

[AWS Essentials: Amazon Machine Images (AMIs)](https://www.youtube.com/watch?v=B7M31vywgs4&ab_channel=LinuxAcademy)

### Introduction

An **Amazon Machine Image (AMI)** is a template that contains a software configuration (operating system, application server, and applications) **required to launch a virtual server in the Amazon EC2 environment**.

### Types of AMI

There are different types of AMIs, each providing a different set of software configurations:

- **Amazon EBS-backed AMIs**: The root device for an instance launched from the AMI is an Amazon Elastic Block Store (EBS) volume created from an EBS snapshot.
- **Instance store-backed AMIs**: The root device for an instance launched from the AMI is an instance store volume created from a template stored in Amazon S3.

### Choosing an AMI

When launching an instance, the AMI determines the following:

- The operating system and its version.
- The root device volume and its size.
- Data volumes, if any.
- Launch permissions that control which AWS accounts can use the AMI to launch instances.
- A block device mapping that specifies the volumes to attach to the instance when it's launched.

### Creating an AMI

Creating an AMI makes it possible to launch multiple instances with the same configuration:

1. Launch an instance and customize it.
2. Create an image of the instance while it's running or stopped.
3. Register the image as an AMI.

### Sharing an AMI

You can share an AMI with specific AWS accounts without making it public. The accounts you share with can launch instances using your AMI.

### Copying an AMI

You can copy an AMI within the same region or to different regions. When you copy an AMI, the resulting AMI is owned by the account that made the copy.

### Deregistering an AMI

When you're finished with an AMI, you can deregister it. Deregistering an AMI doesn't delete the snapshot that was created for the root volume of the instance during the AMI creation process, but it does remove the reference to the AMI.

### Conclusion

AMIs are a fundamental component in the Amazon EC2 instance launch process. They define the initial state of the instance, serve as templates for creating new instances, and can be shared among different AWS accounts or copied across regions.
