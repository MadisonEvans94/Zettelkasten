#incubator 

see here https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/AWSHowTo.iam.html
<a name="introduction"></a>
### Introduction to Amazon Elastic Beanstalk

Amazon Elastic Beanstalk is a fully managed service from Amazon Web Services (AWS) that makes it easy for developers to deploy and run applications in multiple languages without worrying about the underlying infrastructure. 

With Elastic Beanstalk, you can quickly deploy your application with the following AWS services already set up: EC2 for compute, S3 for storage, SNS for notifications, CloudWatch for metrics, and optionally RDS for databases, among other services.

You simply upload your application, and Elastic Beanstalk handles capacity provisioning, load balancing, auto-scaling, and application health monitoring.

<a name="usecases"></a>
### Use Cases

1. **Web Application Hosting**: You can host your web applications on Elastic Beanstalk. Supported platforms include Java, .NET, PHP, Node.js, Python, Ruby, Go, and Docker.

2. **API Backend**: Elastic Beanstalk can also serve as a backend for API calls. You can use it with API Gateway to manage, monitor, and secure your APIs.

3. **Microservices**: Microservices architectures can be easily managed and deployed with Elastic Beanstalk.

<a name="gettingstarted"></a>
### Getting Started

To get started with Elastic Beanstalk, you need to:

1. **Create an AWS Account**: Sign up for an AWS account if you don't have one already.

2. **Install AWS CLI and EB CLI**: Install and configure the AWS Command Line Interface (CLI) and the Elastic Beanstalk Command Line Interface (EB CLI) for your operating system.

3. **Create an Application**: In the Elastic Beanstalk console, choose 'Create a new application' and follow the prompts.

<a name="deployingapplications"></a>
### Deploying Applications

1. **Create Application Version**: After creating an application, you can create an application version. In the Elastic Beanstalk console, go to 'Application versions' and choose 'Create a new version'.

2. **Deploy Application Version**: You can then deploy this application version. Go to 'Environment overview' and choose 'Upload and deploy'.

<a name="managingapplications"></a>
### Managing and Modifying Applications

You can manage your applications using either the Elastic Beanstalk console, the AWS CLI, or the EB CLI. You can modify configurations, such as the type of instances used, the database attached, environment variables, and much more.

<a name="cleaningup"></a>
### Cleaning Up and Deleting Applications

Cleaning up is straightforward:

1. **Delete Environment**: In the Elastic Beanstalk console, navigate to the 'Environment overview' page. Choose 'Actions' and then 'Terminate environment'. 

2. **Delete Application**: To delete an application, navigate to the 'Application overview' page. Choose 'Actions' and then 'Delete application'. 

Please note: Be careful with this step. Once deleted, your environment and application data can't be recovered.

This guide provides an overview of Amazon Elastic Beanstalk. For more detailed information, please refer to the [official AWS Elastic Beanstalk documentation](https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/Welcome.html).


---

### Environments 

An **environment** is a version of your application that is running in AWS Elastic Beanstalk. It consists of an application version and an environment configuration, along with the AWS resources that are provisioned to host your application (like Amazon EC2 instances, Auto Scaling groups, and Amazon RDS DB instances). *For more detailed information, please refer to the* [official AWS Elastic Beanstalk documentation](https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/Welcome.html).

#### Why Use an Environment? 

Environments in Elastic Beanstalk are useful to isolate parts of your application, for instance, to separate development, testing, and production stages of your application. Each environment runs only one application version at a time, and you can run the same version across multiple environments.

#### Types of Environments 

In AWS Elastic Beanstalk, you can create either a **web server** environment or a **worker** environment based on your application needs.

##### Web Server Environment: 

This is meant for hosting web applications or a [[REST API]]. These could be standard websites, micro-services, or backend systems that communicate via HTTP(S) protocols. 

You typically choose this environment when you need to respond to HTTP(S) requests from clients like web browsers or mobile devices. It includes an HTTP(S) load balancer (ELB or ALB based on your choice) to handle client requests and distribute them to your application instances.

##### Worker Environment: 

This environment type is designed for long-running tasks or jobs that you'd typically place in a queue for processing. This could be anything from **sending bulk emails**, **processing files**, or handling jobs that take more than a few seconds to finish.

Worker environments read messages from an [[Amazon SQS (Simple Queue Service)]] queue and then process them. The worker processes the messages asynchronously, which means that the sender doesn't wait for a response before moving on to the next task.

##### When to Choose Which? 

you'd choose a **web server environment** for handling real-time web traffic and a **worker environment** for processing time or resource-intensive tasks in the background. Keep in mind that many applications may require both types of environments, working together. *For example...* a web server might accept image upload requests and then offload the actual image processing to a worker.

#### Creating an Environment

To create an environment in Elastic Beanstalk:

1. Open the Elastic Beanstalk console.

2. In the region selector, choose the AWS Region where you want to create the environment.

3. On the Elastic Beanstalk home page, choose 'Create a new environment'.

4. Choose the environment type and follow the prompts.

---

### Elastic Beanstalk Pricing Guide

Understanding its pricing can be a bit complex due to the variety of services it uses under the hood. This guide will provide you with a detailed understanding of Elastic Beanstalk's pricing and how to estimate costs.


#### 1. Basic Pricing

Elastic Beanstalk itself doesn't cost anything extra. You pay for the AWS resources (like **EC2** instances or **S3** buckets) that your application uses. These costs will vary based on the resources your application requires to run effectively.

The most common resources include:

- **EC2 instances**: These are the virtual servers where your application runs. The cost depends on the type and size of the instances.
- **EBS volumes**: This is the storage for your instances. Again, the cost depends on the amount and type of storage you use.
- **S3 buckets**: Elastic Beanstalk stores your application versions in S3, which incurs a fee.
- **Data transfer**: There's usually a cost for data transferred out of AWS. Data transferred between AWS services in different regions also incurs a cost.

#### 2. Pricing Model

The pricing model of AWS Elastic Beanstalk is based on a **pay-as-you-go** model. This means that you pay for what you use without upfront or long-term commitments. You can start or stop using the service at your convenience, and only pay for what you use.

#### 3. Cost Optimization

To optimize costs, you can take advantage of the following:

- **Reserved Instances**: If you know your application's requirements in advance, you can reserve instances for 1 or 3 years, which can lead to significant savings.
- **Spot Instances**: These are spare EC2 instances that you can bid for, and they can be much cheaper than regular instances.
- **Auto Scaling**: This allows your application to scale resources based on demand, which can both improve performance and lower costs.

#### 4. Pricing Calculator

To estimate your Elastic Beanstalk costs, you can use the [AWS Pricing Calculator](https://calculator.aws/#/). This tool lets you model your expected usage of AWS services and will provide an estimate of the cost. Remember to include all the resources your Elastic Beanstalk application will use.

#### 5. Free Tier

If you're new to AWS, you can take advantage of the [AWS Free Tier](https://aws.amazon.com/free/). The Free Tier offers a generous amount of usage of many AWS services for free. Note that EC2 instances, which Elastic Beanstalk often uses, are part of the Free Tier.

#### 6. Pricing Example

*For example*, if your Elastic Beanstalk application uses the following resources:

- **t3.medium EC2 instances**: The cost will vary based on the region, but let's say it's $0.0416 per Hour.
- **General Purpose SSD (gp2) EBS volumes**: Again, the cost varies, but let's say it's $0.10 per GB-month.
- **Data Transfer**: The first GB is free, and then it's $0.09 per GB up to 10 TB.

Then, the estimated monthly cost would be:

- EC2: $0.0416 * 24 hours * 30 days = $29.95
- EBS: $0.10 * 20GB = $2.00
- Data Transfer: Assuming 100GB out, the cost would be $0.09 * 99GB = $8.91

So, the total estimated cost would be $29.95 (EC2) + $2.00 (EBS) + $8.91 (Data Transfer) = **$40.86** per month.

Please note, these are just rough estimates. The actual cost can vary based on your application's resource usage and other factors.

#### how do I cancel an elastic beanstalk instance?

*I've tested it and verified that it works, but I'm not ready for production yet and want to turn it off for now so that I don't get any unwanted costs*

To stop incurring charges for AWS Elastic Beanstalk, you can **terminate** your environment. This will delete all AWS resources associated with the environment, like EC2 instances, DB instances, S3 buckets, etc. Here's how to do it:

1. Open the Elastic Beanstalk console.
2. Navigate to the management page for your environment.
3. Choose **Actions**, and then choose **Terminate Environment**.
4. In the Terminate Environment dialog box, type the name of your environment to confirm, and then choose **Terminate**.

*Please note*, the termination process may take a few minutes. After it's finished, you should see a message in the environment events indicating that the environment was successfully terminated.

*Also note* that even though the environment is terminated, the application and its versions will still exist in the Elastic Beanstalk, and you can create a new environment for them anytime. If you do not plan to use these in the future, consider deleting them to avoid any S3 storage costs.

*It's recommended* to make sure you have a backup of your environment and related data before terminating the environment. Once the environment is terminated, you won't be able to retrieve any associated data or configurations. 

*Remember* to also delete any other unneeded resources in your AWS account to avoid unexpected charges. These could include EBS volumes, Elastic IP addresses, and other resources you may have created separately.