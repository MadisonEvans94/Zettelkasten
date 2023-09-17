#TODO
 
## Introduction

**Amazon's Identity and Access Management (IAM)** is a service that helps you securely control access to AWS resources. You use IAM to control who is **authenticated** (signed in) and **authorized** (has permissions) to use resources.

## What is IAM? <a name="what-is-iam"></a>

IAM lets you manage access to your AWS resources. With IAM, you can create **users**, **groups**, and **roles** to which you can grant permissions to access AWS resources.

## Users, Groups, and Roles in AWS IAM
<a name="users"></a>
### IAM Users

An IAM **user** is an identity within your AWS account that has specific custom permissions *(for example, permissions to create an EC2 instance, or to read an object in an S3 bucket)*. You can use a user to sign in to the AWS Management Console, make direct AWS API calls, or to access resources, like deploying applications onto AWS services.

Typically, you create an IAM user for **each person** who needs access to your AWS account. Users consist of a name and credentials, which could be a password for AWS Management Console access or an access key for API calls.

<a name="groups"></a>
### IAM Groups

An IAM **group** is essentially a collection of IAM users. It cannot be identified as a 'principal' in a policy, and you cannot sign in as a group. Its sole function is to **manage permissions collectively** for multiple users. If you assign a permission to a group, all users in that group receive that permission.

*For example, you could have a group called "`Developers`" and give it the types of permissions that developers typically need. Any user in that group would have the permissions assigned to the group*.

<a name="roles"></a>
### IAM Roles

An IAM **role** is similar to a user, in that it is an identity with permission policies that determine what the identity can and cannot do in AWS. However, a role does not have standard long-term credentials (password or access keys) associated with it. Instead, when you assume a role, it provides you with temporary security credentials **for your role sessions**. 

Roles are used when you want to delegate access with specific permissions to trusted entities, such as IAM users in another account, applications running on an EC2 instance, or an AWS service like AWS Lambda.

This separation of duties allows you to operate in a more secure environment, where not all operations are conducted with a 'root' level of access.

For more information about IAM users, groups, and roles, refer to the [official AWS IAM documentation](https://docs.aws.amazon.com/IAM/latest/UserGuide/id.html).


### IAM Permissions
- Users or Groups can be assigned json documents called policies
- these policies define what they are *allowed* access to 
- [ ] provide example json

https://www.udemy.com/course/aws-certified-cloud-practitioner-new/learn/lecture/20281863#overview

## Why use IAM? <a name="why-use-iam"></a>

IAM allows you to manage users and their level of access to the AWS console. It is a feature of your AWS account offered at no additional charge. It provides:

### Shared access to your AWS account: 
You can grant other people permission to administer and use resources in your AWS account without having to share your password.

### Granular permissions: 
You can grant different permissions to different people for different resources.

### Secure access to applications running on EC2 instances: 
You can use IAM roles to manage credentials for applications that run on EC2 instances.

## Key Concepts <a name="key-concepts"></a>

### IAM Users
An IAM user is an entity that you create in AWS to represent the person or application that uses it to interact with AWS.

### IAM Groups
An IAM group is a collection of IAM users. You can use groups to specify permissions for multiple users, which can make it easier to manage the permissions for those users.

### IAM Policies
IAM policies define permissions for an action regardless of the method that you use to perform the operation. For example, if a policy allows the `GetUser` action, then a user with that policy can get user information from the AWS Management Console, the AWS CLI, or the AWS API.

### IAM Roles
An IAM role is similar to a user, in that it is an AWS identity with permission policies that determine what the identity can and cannot do in AWS. However, a role does not have standard long-term credentials (password or access keys) associated with it. Instead, when you assume a role, it provides you with temporary security credentials for your role session.

## Best Practices <a name="best-practices"></a>

1. **Grant least privilege**: Only grant the permissions required to perform a task.

2. **Rotate credentials regularly**: Change your own passwords and access keys regularly, and make sure that all IAM users in your account do the same.

3. **Remove unnecessary credentials**: Passwords and access keys that are not needed should be removed.

4. **Use groups to assign permissions**: Assign permissions to user groups. It's more efficient than assigning permissions to users individually.

5. **Enable MFA for privileged users**: Enable multi-factor authentication (MFA) for all of your users.

## Getting Started <a name="getting-started"></a>

You can create and manage AWS IAM roles, users, groups, and permissions via the AWS Management Console, AWS CLI, or AWS SDKs.

### **Creating a user in IAM**:
1. Open the IAM console.

2. In the navigation pane, choose Users, and then choose Add user.

3. Type the user name for the new user. This is the sign-in name for AWS.

4. Select the type of access this set of users will have.

5. Choose Next: Permissions.

6. Specify how you want to assign permissions to this set of new users.

7. Choose Next: Tags.

8. (Optional) Add metadata to the user by attaching tags.

9. Choose Next: Review to see all of the choices you made up to this point.

10. When you are ready to proceed, choose Create user.


## Advanced IAM Policies <a name="advanced-policies"></a>

IAM Policies are JSON documents that when associated with an identity or resource define their permissions. IAM Policies can be classified as:

### Identity-based policies: 
These are attached to an identity (user, group of users, or roles) and allow or deny permissions to AWS resources.

### Resource-based policies: 
These are attached to a resource (e.g., Amazon S3 bucket, Amazon SNS topic, etc.) and they allow or deny the specified principal (account, user, role, federated user) permissions to the specific resource to which they're attached.

## Additional Resources <a name="additional-resources"></a>

- [IAM Documentation](https://docs.aws.amazon.com/IAM/latest/UserGuide/introduction.html)
- [IAM FAQs](https://aws.amazon.com/iam/faqs/)
- [IAM Best Practices](https://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices.html)
