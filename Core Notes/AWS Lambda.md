#incubator 
upstream: [[AWS]]

This document provides a comprehensive understanding of **AWS Lambda**, a server-less compute service provided by **Amazon Web Services (AWS)** that lets you run your code without provisioning or managing servers.

### What is AWS Lambda?

**AWS Lambda** is a service that lets you run your applications and services without worrying about provisioning or managing servers. With Lambda, you can run code for virtually any type of application or backend service, all with zero administration. Just upload your code and Lambda takes care of everything required to run and scale your code with high availability.

### Key Features of AWS Lambda

#### 1. **No Server Management**: 
AWS Lambda automatically runs your code without requiring you to provision or manage servers.

#### 2. **Continuous Scaling**: 
AWS Lambda automatically scales your applications in response to incoming request traffic.

#### 3. 
**Sub-second Metering**: Billing is metered in increments of 100 milliseconds, making it cost-effective to run even small, quick functions.

#### 4. **Event-Driven Execution**: 
AWS Lambda can be set up to trigger from over 140 AWS services or called directly from any web or mobile application.

#### 5. **Integrated Security Model**: 
With AWS Lambda, you can set up your code to automatically trigger from other AWS services, call it directly from any web or mobile app, or use it as the backend for Alexa skills.

#### 6. **Built-in Fault Tolerance**: 
AWS Lambda has built-in fault tolerance. It maintains compute capacity across multiple Availability Zones in each region to help provide both high availability and ensure consistent performance.

### Use Cases for AWS Lambda

- **Real-Time File Processing**: You can use AWS Lambda to execute code when changes to objects occur in Amazon S3 buckets, making it suitable for ETL jobs, generating thumbnails, etc.

- **Real-Time Stream Processing**: You can use AWS Lambda to process, filter and route streaming data for application activity tracking, transaction order processing, real-time analytics, and more.

- **Web Applications**: AWS Lambda can be used to host [[REST API]] APIs, website backends, or mobile backends.

- **IoT Backends**: AWS Lambda can be used to validate, transform and transfer device data to other devices or backend services.

### AWS Lambda Workflow

#### 1. **Create a Lambda Function**: 
The first step is to create a Lambda function. You can do this using the AWS Management Console, AWS CLI, or AWS SDKs.

#### 2. **Upload Your Code**: 
You upload your application code in the form of a deployment package to the Lambda function. The package can be a .zip file containing your code and any dependencies.

#### 3. **Set Up Event Source**: 
You set up an event source that triggers the Lambda function. The event source can be an AWS service such as **Amazon S3**, **Amazon DynamoDB**, **Amazon SNS**, etc.

#### 4. **Invoke Your Function**: 
AWS Lambda executes your function on your behalf when the event source is triggered.

#### 5. **Lambda Function Does Its Job**: 
The Lambda function runs and processes the event. It then shuts down.

#### 6. **Monitor & Debug**: 
AWS Lambda automatically monitors Lambda functions on your behalf and reports metrics through Amazon CloudWatch. To help debug any issues with your Lambda function, Lambda logs all the request events and the results to CloudWatch Logs.


### Pricing

With AWS Lambda, you pay only for what you use. You are charged based on the number of requests for your functions and the time your code executes

- Free tier: 1M free requests per month and **400,000 GB-seconds** of compute time per month.
- **$0.20** per 1 million requests thereafter (**$0.0000002 per request**)
- **$0.0000166667** for every GB-second

The amount of memory you allocate to your function determines the cost, so by configuring your memory setting, you can balance performance and cost.

*Remember to always refer to the [official AWS Pricing page](https://aws.amazon.com/lambda/pricing/) for the most accurate and up-to-date information.*

### Limitations

While AWS Lambda offers a powerful, server-less environment, it's important to know its limitations:

#### 1. **Time Limit**: 
The maximum execution duration per request is **15 minutes**.

#### 2. **Memory Allocation**: 
You can allocate between **128 MB** and **10,240 MB** to your Lambda function in **1-MB** increments.

#### 3. **Deployment Package Size**: 
The deployment package size is limited. It's **50 MB** (compressed, .zip/.jar file) for direct upload and **250 MB** (uncompressed, including layers) for deployment.

*Please refer to the [official AWS Lambda documentation](https://docs.aws.amazon.com/lambda/latest/dg/gettingstarted-limits.html) for a comprehensive list of service quotas and limitations.*

### Conclusion

**AWS Lambda** is a powerful service that lets you run your code without provisioning or managing servers. Its ability to automatically scale with incoming traffic, coupled with an integrated security model and high fault tolerance, makes it an excellent choice for many use cases. Understanding its pricing model and limitations can help you design and develop effective, cost-efficient serverless applications.