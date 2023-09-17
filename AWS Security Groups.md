#incubator 
upstream: [[AWS]]

---

**video links**: 

---

## Introduction to Security Groups

Two key concepts within AWS are **Security Groups** and **VPC (Virtual Private Cloud)**. Understanding these concepts are crucial to effectively manage the security and networking of your AWS environment.

Security Groups act as a virtual firewall for your instance to control inbound and outbound traffic. When you launch an instance in a VPC, you can assign up to five security groups to the instance. Security Groups are stateful — if you send a request from your instance, the response traffic for that request is allowed to flow in regardless of inbound security group rules.

In more technical terms, security groups control access to instances using ingress (inbound) and egress (outbound) rules. By default, *security groups allow all outbound traffic but deny all inbound traffic*. 

### Key points:

- **Inbound rules**: Controls the incoming traffic to your instances. By default, no inbound traffic is allowed until you add inbound rules to the security group.

- **Outbound rules**: Controls the outgoing traffic from your instances. By default, an instance can send traffic to any destination.

- **Stateful**: If you send a request from your instance, the response traffic for that request is allowed to flow in regardless of inbound security group rules.

[[Understanding VPC with an Analogy]]

### Examples: 
Here's an example of what an **inbound rule** might look like:

- **Type**: SSH (Secure Shell)
- **Protocol**: TCP
- **Port Range**: 22
- **Source**: 203.0.113.1

This rule would allow an instance with the IP address 203.0.113.1 to connect to your EC2 instance over SSH.
And here's an example of an **outbound rule**:

- **Type**: HTTP
- **Protocol**: TCP
- **Port Range**: 80
- **Destination**: 0.0.0.0/0

This rule would allow your EC2 instance to send HTTP traffic to any destination.

## Understanding Security Groups
Security groups act like a firewall around your Amazon EC2 instances, helping regulate inbound (incoming) and outbound (outgoing) traffic. When you set up a new EC2 instance, you have the opportunity to assign it to existing security groups or create a new one. 

### Terms Explained
Let's simplify some of the terms first:

- **Security Group**: Think of this like the bouncer for a club. It decides who can come in and go out. 

- **Inbound rules**: These are the rules that decide who can come into the club. If a rule doesn't exist for them, they can't come in!

- **Outbound rules**: These rules decide who can leave the club. By default, everyone inside the club can leave.

- **Stateful**: This is a cool feature. Imagine if you are in the club and you make a phone call (send a request). The bouncer will let the person (response traffic) on the other side come in, no matter the inbound rules.

### Best Practices
Now that we've defined the terms, let's dive into best practices for security groups:

1. **Principle of Least Privilege**: Only open up what you absolutely need. If your application doesn't need to accept traffic on a certain port, don't allow it in your security group.

2. **Don't Rely on Default Security Groups**: It's tempting to use the default security group that comes with your AWS account, but this group often has permissive settings, which can be a security risk.

3. **Limit Open Ports**: The more ports you leave open, the more potential entry points you create for unwanted traffic.

4. **Specify Restricted IP Ranges**: Rather than allowing any IP address (0.0.0.0/0) to connect to your instances, specify the IPs or ranges as narrowly as possible.

5. **Regular Audits**: Regularly review your security group rules and remove any rules that are no longer necessary.

6. **Avoid Using Single Security Group for All Instances**: It might be easier to manage one security group for all instances, but it's more secure to customize each security group according to the needs of the specific instance.

7. **Document Changes**: Make sure you document any changes made to the security groups. This is helpful in case you need to backtrack or understand why a specific change was made.

Remember, security on AWS is a shared responsibility. While AWS manages the security of the cloud, customers are responsible for security in the cloud—including deciding who has access to your instances.