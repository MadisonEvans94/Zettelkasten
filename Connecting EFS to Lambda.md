#seed 
upstream:

---

**video links**:  

---
Additional Questions: 
- what exactly is an access point? 
- why does efs need to be created in vpc? 
- why do we need to open port 2049 on the access point?
- 

![[Screen Shot 2023-06-29 at 12.37.42 PM.png]]

see [[AWS Security Groups]] and

> start here (Mount efs on instance ): https://www.youtube.com/watch?v=4cquiuAQBco&t=0s&ab_channel=SrceCde
> 
> then here: https://www.youtube.com/watch?v=FA153BGOV_A&t=0s&ab_channel=SrceCde

Based on the video "How to use EFS (Elastic File System) with AWS Lambda" by Srce Cde, here is a high-level step-by-step guide:

1. **Create a Security Group**: This can be done from the EC2 or VPC console in AWS. The security group is created without any inbound rules but with an outbound rule for all traffic.

2. **Create an Elastic File System (EFS)**: Navigate to the EFS management console in AWS and create a new file system. Configure the network access, select the VPC (default or custom), and select the security group created in the previous step. Enable lifecycle management and encryption as needed.

3. **Add Access Points within EFS**: Add an access point to the EFS, providing a name, POSIX user, and directory path. Set the owner user ID and permissions as needed.

4. **Create an IAM Role for Lambda Permissions**: In the IAM management console, create a new role for the Lambda service. Attach the necessary permissions, such as `lambda_execute`, `AWSLambdaVPCAccessExecutionRole`, and `AmazonElasticFileSystemClientFullAccess`.

5. **Create a Lambda Function**: In the Lambda management console, create a new function. Select the runtime (e.g., Python 3.8) and use the existing role created in the previous step.

6. **Configure VPC within the Lambda Function**: In the Lambda function settings, add the VPC, select the subnets, and select the security group created in the first step.

7. **Add Elastic File System to the Lambda Function**: In the Lambda function settings, add the file system. Select the EFS and access point created in the second and third steps, and configure the local mount path.
--- 

8. **Update Lambda Function Code for Testing EFS Integration**: Update the code of the Lambda function to test the EFS integration. The code should import the OS module, list the directory before and after creating a file in the EFS, and print the results.

9. **Open Port 2049 in Security Group**: Edit the inbound rules of the security group created in the first step to add a custom TCP rule for port 2049, allowing access from anywhere.

10. **Test the EFS Integration with the Lambda Function**: Run the Lambda function and check the logs to confirm that the EFS integration is working correctly.

Please note that this is a high-level guide and the specific details and parameters may vary based on your specific use case and AWS configuration.