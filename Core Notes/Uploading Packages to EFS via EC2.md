
#incubator 
upstream: [[EC2]], [[EFS]]

---

**video links**: 

---

## Introduction
This guide will walk you through the process of loading Python packages into your **EFS (Elastic File System)** instance via an **EC2 (Elastic Compute Cloud)** instance.

>Note: all the steps from updating the package lists and installing necessary utilities to mounting the EFS filesystem, installing Python packages and finally un-mounting the filesystem should be performed while SSH'd into your EC2 instance. 

---

## Can this be done from the AWS Management Console or does it have to be done remotely through terminal? 

The process described mostly involves commands that need to be executed within a shell, and for that, you need to use a terminal (command line interface) on your local machine to SSH into the EC2 instance.

The AWS Management Console, on the other hand, is a web-based interface for managing and monitoring your AWS resources. While it provides ways to manage EC2 instances (start, stop, terminate, etc.) and EFS volumes, it doesn't provide a way to SSH into an EC2 instance or execute shell commands directly on the EC2 instance.

*see [[Connecting to EC2 via ssh]] for more details* 

---

## Will I have issues with permissions access? 

The **"0755"** permission code means that the owner of the file (or directory) has read, write, and execute permissions, while the group and others have only read and execute permissions. In this context, it indicates that only the owner of the directory (the user under whose account the directory was created) can write to it, i.e., create, delete, or modify files in it.

If the EC2 instance and the EFS share the same owner (the same user ID), then you should have no issues following the guide, because the EC2 instance will have write permissions to the EFS. 

If the EC2 instance doesn't share the same owner as the EFS, you will encounter permission issues. In such a case, you will need to either modify the permissions of the EFS to allow the EC2 instance to write to it (which could potentially have security implications), or you can ensure that the EC2 instance is running under the same user ID as the EFS owner. 

However, by default, the user ID when you log into an EC2 instance is "ec2-user", which may not have the necessary permissions. If you need to run commands as root (user ID 0), you can use `sudo` before the commands that require root permissions.

## Procedure
For this workflow walkthrough, we will be running a scenario of installing python and other related packages, but this workflow can apply to other dependencies as well 

### 1. Make sure your EC2 instance has all necessary security groups and roles 

### 1. Install Necessary Utilities <a name="install-necessary-utilities"></a>

First, you need to update the package lists for upgrades and new package installations.

```bash
sudo yum update -y
```

Then, install the Amazon EFS utilities.

```bash
sudo yum install -y amazon-efs-utils
```

### 2. Mount the EFS File System <a name="mount-the-efs-file-system"></a>

Next, create a directory to mount the file system.

```bash
sudo mkdir /mnt/efs
```
- [ ] TODO 
```bash 
sudo mount -t efs -o tls,accesspoint=fsap-0621054da84634ea6 fs-01ea000f159df1364:/ /mnt/efs

```

Mount your EFS file system using the command below. Replace `fs-0dbe4d477f6f45b26` with your actual EFS ID.

```bash
sudo mount -t nfs -o nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport fs-01ea000f159df1364.efs.us-east-2.amazonaws.com:/ /mnt/efs
```

Verify that your EFS file system has been mounted correctly by checking the list of mounted file systems.

```bash
df -h
```

### 3. Install Python and Packages <a name="install-python-and-packages"></a>
numpy-PIL-sklearn-dependencies
You can now install Python3 and pip (if they aren't already installed).

```bash
sudo yum install python3
```

Next, create a directory in your EFS instance for the Python packages.

```bash
sudo mkdir /mnt/efs/python
```

Use pip to install the desired packages into this directory. In this example, we are installing numpy, Pillow, and scikit-learn.

```bash
pip3 install --target=/mnt/efs/python Pillow scikit-learn
```

#### In case you get `-bash: pip3: command not found`

In most cases, `pip3` should be installed along with Python 3 when you use `yum install python3`. However, if it's not installed for some reason, you can install it separately.

Here's how you can install `pip3` on an Amazon Linux or Amazon Linux 2 instance:

```bash
sudo yum install python3-pip
```

After running this command, `pip3` should be installed and you should be able to use it to install Python packages. You can verify the installation by running:

```bash
pip3 --version
```

This should print the version of `pip3` that's installed. Once `pip3` is installed, you can proceed with installing the Python packages on your EFS file system.

>Note: When you use the `--target` option with `pip install`, it specifies the directory where the packages will be installed.

...So, by running `pip3 install --target=/mnt/efs/python numpy Pillow scikit-learn`, you are instructing pip to install the numpy, Pillow, and scikit-learn packages into the `/mnt/efs/python` directory on your EFS file system.

When your Lambda function runs, it will be able to access these packages by adding `/mnt/efs/python` to the system path. This is done by the following lines in your Lambda function:

```python 
import sys 
sys.path.insert(0, "/mnt/efs/python")
```

After these lines, you should be able to import `numpy`, Pillow, and `scikit-learn` in your Lambda function just like any other Python package.


### 4. Clean Up <a name="clean-up"></a>

Once you have completed the previous steps, unmount the EFS file system.

```bash
sudo umount /mnt/efs
```

Then, you can exit the EC2 instance.

```bash
exit
```

>To avoid unnecessary charges, terminate the EC2 instance from the EC2 dashboard.

With the Python packages `numpy`, `Pillow`, and `sklearn` installed in your EFS instance, they are now ready to use in your Lambda function. Remember to include "`/mnt/efs/python`" to your Python path in your Lambda function code:

```python
import sys
sys.path.insert(0, "/mnt/efs/python")

# Now you can import numpy, PIL and sklearn
import numpy
from PIL import Image
import sklearn
```

>Note: These instructions are based on a standard Linux distribution like Amazon Linux 2 and may need adjustment depending on your specific setup.