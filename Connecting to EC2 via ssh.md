#seed 
upstream: [[EC2]]

---

**video links**: 

[# SSH to EC2 Instances using Linux or Mac Tutorial](https://www.youtube.com/watch?v=8UqtMcX_kg0&ab_channel=StephaneMaarek)

---


This guide will walk you through the process of connecting to your EC2 instance using Secure Shell (SSH).

## 1. Generate a Key Pair <a name="generate-a-key-pair"></a>

A **key pair** is necessary for establishing a secure connection. Follow these steps to generate it in the AWS Management Console:

- Go to the EC2 section of the AWS Management Console. Navigate to the "Key Pairs" section found under the "Network & Security" grouping.
  
- Click "Create key pair".

- Assign a name to your key pair and select "pem" as the file format.

- Click "Create key pair".

The above steps will prompt a download of a `.pem` file. Ensure you keep this file secure as it serves as your **private key**. You'll need it to establish a connection to your instance. Also, remember that you **cannot** download this file again from AWS due to security reasons. Always keep a secure copy.

## 2. Launch Your EC2 Instance <a name="launch-your-ec2-instance"></a>

While launching your EC2 instance, you will be asked to choose a key pair. Select the key pair you just created. If the instance is already up and running, you may need to stop it and change the key pair, or you can use the key pair that was initially chosen at the time the instance was launched.

## 3. Connect to Your EC2 Instance <a name="connect-to-your-ec2-instance"></a>

Now that you have your key pair and EC2 instance ready, you can connect to your instance. Here's how:

### Open a terminal on your local machine. 
If you're on Windows, you can use PuTTY, while on MacOS and Linux, you can use the Terminal app.

### Access your Private Key 
- To ensure SSH works, your key (`.pem` file) should not be publicly viewable. Navigate to the directory where your `.pem` file is saved and run the following command:

  ```bash
  chmod 400 /path/my-key-pair.pem
  ```
  
  The command `chmod 400 /path/my-key-pair.pem` is used to change the file permissions of your private key file (`.pem` file). The number `400` represents the permissions that are being set:

	- The first digit (representing the owner's permissions) is `4`, which stands for "read" permission.
	- The second and third digits (representing the group and others' permissions respectively) are `0`, which means no permissions.

>	Note: the same procedure is done if you have a `.cer` file as opposed to a `.pem` file

```bash
chmod 400 /path/my-key-pair.cer
```

**TLDR**: This command is setting the permissions on the `.pem` file so that the owner (you) can read it, but cannot write to or execute it, and no other users can read, write, or execute it. This is a necessary step because SSH requires that private key files are not accessible by others and should have restrictive permissions set.

>Remember to replace "`/path/my-key-pair.pem`" with the actual path to your `.pem` file.

### Establish a connection to your EC2 instance 
For this, you need the public IP of your instance, which you can find on the EC2 dashboard under "Instances". Use the following command to connect:

  ```
  ssh -i /path/my-key-pair.pem ec2-user@public-ip-of-your-instance
  ```
  
Replace "`/path/my-key-pair.pem`" with the actual path to your `.pem` file and "`public-ip-of-your-instance`" with the actual public IP of your instance.

Congratulations! If all the steps were executed correctly, you should now be connected to your EC2 instance and ready to run commands.

>Note: These instructions are based on a standard Linux distribution like Amazon Linux 2 and may need adjustment depending on your specific setup. Also, the default username here is "ec2-user", but it could vary depending on the AMI (Amazon Machine Image) you use. For instance, if you use an Ubuntu AMI, the default username would be "ubuntu".





