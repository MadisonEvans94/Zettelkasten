#incubator 
upstream: [[AWS]]

This document provides a comprehensive understanding of AWS API Gateway, a fully managed service from Amazon Web Services that makes it easy for developers to create, publish, maintain, monitor, and secure APIs at any scale.

### What is AWS API Gateway

**AWS API Gateway** is a service that makes it easy for developers to create, deploy, manage and secure APIs at any scale. You can create APIs that access AWS or other web services, as well as data stored in the AWS Cloud. API Gateway handles all the tasks involved in accepting and processing up to hundreds of thousands of concurrent API calls, including traffic management, authorization and access control, monitoring, API version management, and more.

### Features of AWS API Gateway

*Here are some of the key features that AWS API Gateway offers:*

#### 1. **Performance at scale**: 
API Gateway handles all the tasks involved in accepting and processing concurrent API calls, including traffic management, data transformations, and more.

#### 2. **SDK Generation**: 
AWS API Gateway can automatically generate client SDKs for a number of platforms, including JavaScript, iOS, and Android.

#### 3. **Security**: 
You can use AWS Identity and Access Management (IAM) and Amazon Cognito to authorize access to your APIs.

#### 4. **API Lifecycle Management**: 
API Gateway provides version management and lifecycle features that help you manage your APIs.

#### 5. **Server-less Integration**: 
API Gateway has out-of-the-box integration with AWS Lambda, allowing you to run your code without provisioning or managing servers.

### AWS API Gateway Workflow

*Here's a simplified example of how a workflow might look in AWS API Gateway:*

#### 1. **Create an API**: 
This is the initial step where you create your API. You can create a RESTful API or a WebSocket API.

#### 2. **Define Resources and Methods**: 
In this step, you define the endpoints (`resources`) and HTTP verbs (`methods`) for your API. For example, you might define a `POST` method on a `/users` resource to create a new user.

#### 3. **Set Up Method Request**: 
This is where you configure how the method request is handled. You can choose to pass through the incoming request, or you can set up request validation and/or request mapping templates to transform the incoming request.

#### 4. **Set Up Integration Request**: 
In this step, you set up how the method request should be sent to the backend. You can integrate with an HTTP endpoint, an [[AWS Lambda]] function, or an AWS service.

#### 5. **Set Up Method Response**: 
This is where you define the structure of the method response. You can define status codes, headers, and the response model.

#### 6. **Set Up Integration Response**: 
Here you define how to transform the backend response into the method response. You can define mapping templates and specify how errors should be handled.

#### 7. **Deploy API**: 
Once you have defined your API, you can deploy it to a stage. A stage is a logical reference to a lifecycle state of your API (e.g., `dev`, `prod`).

*Here is an example of how an API Gateway workflow would look:*

```plaintext
API: MyApi
│
└── /users
    ├── GET
    │   ├── Method Request
    │   ├── Integration Request (Lambda: ListUsers)
    │   ├── Method Response
    │   └── Integration Response
    ├── POST
    │   ├── Method Request
    │   ├── Integration Request (Lambda: CreateUser)
    │   ├── Method Response
    │   └── Integration Response
    └── /{id}
        ├── GET
        │   ├── Method Request
        │   ├── Integration Request (Lambda: GetUser)
        │   ├── Method Response
        │   └── Integration Response
        ├── PUT
        │   ├── Method Request
        │   ├── Integration Request (Lambda: UpdateUser)
        │   ├── Method Response
        │   └── Integration Response
        └── DELETE
            ├── Method Request
            ├── Integration Request (Lambda: DeleteUser)
            ├── Method Response
            └── Integration Response
```

### Pricing

AWS API Gateway pricing is based on the number of API calls received, plus data transfer costs. There is also a cost associated with caching. 

- $3.50 per million API calls received.
- $0.09 per GB of data transferred out.
- Caching price varies based on the cache size.

*Always refer to the [official AWS Pricing page](https://aws.amazon.com/api-gateway/pricing/) for the most accurate and up-to-date information.*

### Limitations

While AWS API Gateway is a powerful tool, it also has certain limitations:

#### 1. **Throttling**: 
By default, the REST APIs in API Gateway are limited to **10,000 requests per second (RPS)**. See [[Throttling]] for details

#### 2. **Timeouts**: 
The integration timeout period can be set to a value between **50 milliseconds** and **29 seconds** for all integration types.

#### 3. **Payload size**: 
The maximum payload size for the POST method is **10 MB**.

#### 4. **Resource policies**: 
AWS API Gateway resource policies are limited to a size of **20480 bytes**.

*Please refer to the [official AWS API Gateway documentation](https://docs.aws.amazon.com/apigateway/latest/developerguide/limits.html) for a comprehensive list of service quotas and limitations.*

### Conclusion

AWS API Gateway is a powerful, fully managed service that allows developers to easily create, deploy, manage and secure APIs at scale. It provides useful features such as SDK generation, security controls, lifecycle management and serverless integration. However, like any service, it comes with its own pricing and limitations, so it's important to understand these when planning your API deployment.