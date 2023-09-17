
#bookmark 
upstream:

---

**video links**: 

[# Create an AWS Cognito User Pool](https://www.youtube.com/watch?v=n3br_TzJW28&ab_channel=BrianMorrison)

---

## Introduction
[[AWS Cognito]] is a user management and authentication service provided by AWS that can be easily integrated with your web and mobile applications. In this tutorial, we will discuss the concept of **User Pools** within AWS Cognito and how they can be created and connected to another AWS service such as **Elastic Beanstalk** or **API Gateway**.

## Definition 

A **User Pool** is essentially a user directory in AWS Cognito. It allows you to manage user **registration** and **authentication** in a secure and scalable manner. Once you've created a User Pool, you can set up web or mobile applications as clients to this pool and let Cognito handle user management for you.

Key features include:

- User registration and authentication.
- User profile management.
- Social media login through federation (Facebook, Google, Amazon).
- SAML Identity provider support for enterprise federation.
- MFA (multi-factor authentication).
- Customizable UI for authentication flows.

## Creating a User Pool
- [ ] brief summary of how to get to user pool console 

#### 1. Configure sign-in experience

##### Authentication providers
- [ ] INFO
- [x] IMAGE

![[Screen Shot 2023-06-26 at 3.21.42 PM.png]]

#### 2. Configure security requirements
##### Password policy
- [x] IMAGE
- [ ] INFO
![[Screen Shot 2023-06-26 at 3.23.06 PM.png]]

##### Multi-factor authentication
![[Screen Shot 2023-06-26 at 3.24.00 PM.png]]
- [ ] INFO
- [x] IMAGE

##### User account recovery
- [ ] INFO
- [x] IMAGE
![[Screen Shot 2023-06-26 at 3.25.19 PM.png]]


#### 3. Configure sign-up experience
##### Self-service sign-up
- [ ] INFO
- [x] IMAGE
![[Screen Shot 2023-06-26 at 3.25.55 PM.png]]

##### Attribute verification and user account confirmation
- [ ] INFO
- [x] IMAGE

![[Screen Shot 2023-06-26 at 3.26.27 PM.png]]

##### Required attributes
- [ ] INFO
- [x] IMAGE
![[Screen Shot 2023-06-26 at 3.27.21 PM.png]]

#### 4. Configure message delivery
[[AWS SES]]
*see [[service roles in AWS]] for more*
##### Email
- [ ] INFO
- [x] IMAGE
![[Screen Shot 2023-06-26 at 3.28.08 PM.png]]

##### SMS
- [ ] INFO
- [x] IMAGE
![[Screen Shot 2023-06-26 at 3.28.27 PM.png]]

#### 5. Integrate your app
##### User pool name
- [ ] INFO
- [x] IMAGE
![[Screen Shot 2023-06-26 at 3.29.40 PM.png]]

##### Hosted authentication pages
- [ ] INFO
- [x] IMAGE
![[Screen Shot 2023-06-26 at 3.30.02 PM.png]]

##### Initial app client
- [ ] INFO
- [x] IMAGE
![[Screen Shot 2023-06-26 at 3.30.30 PM.png]]

###### Advanced app client settings
###### Attribute read and write permissions



---



## Connecting User Pool to AWS Services

### 1. Elastic Beanstalk

To use Cognito User Pool with an application deployed on Elastic Beanstalk:

1. **Get your User Pool ID and App Client ID**: You will need these values when configuring your application to use Cognito for authentication.

2. **Configure your application**: This depends on your application's language/framework. AWS provides SDKs for many languages such as JavaScript, Python, Java, .NET, and more. You need to integrate AWS SDK in your application, and use the Cognito functions for registration, login, and user management.

3. **Update environment variables in Elastic Beanstalk**: Go to your Elastic Beanstalk environment, navigate to 'Software' in the configuration and add your User Pool ID and App Client ID as environment variables. 

### 2. API Gateway

To protect an API Gateway with a Cognito User Pool:

1. **Navigate to API Gateway from the AWS Console**.

2. **Select the API you want to secure**.

3. **Select a method (e.g., GET, POST)**. Under 'Method Request', you can set up authorization.

4. **Choose 'AWS_IAM' for Authorization** and select the previously created User Pool for the 'OAuth Scopes'. 

5. **Deploy your API**: Navigate back to the 'Resources' page and click on 'Actions' -> 'Deploy API'.

6. **Update your application**: Ensure that your application requests include an Authorization header with a valid token obtained from Cognito.

## Conclusion

Cognito User Pools are a powerful way to manage user identities in your application. They integrate well with other AWS services and offer a variety of features to support common user management tasks. Always ensure you review the default settings and adjust them to your specific needs and the needs of your application.