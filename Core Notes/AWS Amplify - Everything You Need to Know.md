#incubator 
upstream: [[AWS]]

---

**video links**: 

---

# AWS Amplify - Everything You Need To Know

AWS Amplify is a powerful development platform from Amazon Web Services (AWS) designed to simplify the creation, deployment, and management of cloud-powered mobile and web applications. It provides developers with an extensive range of tools and services.

This comprehensive guide covers everything you need to know about AWS Amplify, including best practices for auth flows.


## Introduction to AWS Amplify
**AWS Amplify** is designed to enable developers to build full-stack applications by leveraging various AWS services. It includes libraries to connect your app to AWS services, a CLI to manage your AWS resources, and a hosting service for deploying web apps.

## Installation and Configuration

To install Amplify, you'll need to have Node.js, npm, and Git installed on your machine. Run the following command:

```bash
npm install -g @aws-amplify/cli
```

To configure Amplify, you need to connect it to your AWS account. Run:

```bash
amplify configure
```

Follow the prompts to sign into your AWS account, specify the AWS region, and create a new IAM user.

## Amplify CLI

The **Amplify Command Line Interface (CLI)** is a unified toolchain to create, integrate, and manage the AWS cloud services for your app. 

Here's how to initialize a new Amplify project:

```bash
amplify init
```

This command will prompt you for some information about your project and AWS setup.

## Authentication

Amplify's Auth category provides authentication features. It's powered by AWS Cognito and it handles user management and authentication functions. To add auth to your application, run:

```bash
amplify add auth
```

To configure advanced settings, choose "Manual configuration".

After you've set up the auth, you need to push your changes to the cloud:

```bash
amplify push
```

You can retrieve the current authenticated user information with `Auth.currentAuthenticatedUser()`.

> see [[Authentication with Amplify]] for more details 



## APIs (REST & GraphQL)

Amplify's API category provides an interface for retrieving and persisting your data via REST or GraphQL. 

Here's how to add a GraphQL API:

```bash
amplify add api
```

Select "GraphQL", provide an API name, and choose an authorization type (for most use cases, "API key" will be sufficient). 

## Storage

Amplify's Storage category provides a simple mechanism for managing user content in public, protected, or private storage buckets.

Here's how to add storage:

```bash
amplify add storage
```

This will prompt you to choose between "Content (Images, audio, video, etc.)" and "NoSQL Database". For file storage, choose "Content".

## Functions (Lambda)

Amplify's Function category allows you to manage and update AWS Lambda functions. To add a function, run:

```bash
amplify add function
```

This will guide you through setting up a new Lambda function.

## Hosting

Amplify Console offers a Git-based workflow for deploying and hosting your app. Every code commit triggers a build and deploy process. 

To add hosting to your project, run:

```bash
amplify add hosting
```

Choose "Amplify Console" (recommended for fullstack projects).

## Best Practices

### 1. Version Control: 
Use version control systems like Git and ensure the `amplify` directory is included in your version control. The `amplify/backend` directory contains your backend cloud resource configurations.

### 2. Environment Management: 
Amplify supports multiple environments. Use `amplify env` commands to manage your environments. This feature enables you to separate your development, staging, and production environments.

### 3. Auth Flows: 
Consider your user experience and security requirements when designing auth flows. If your app involves sensitive data, consider adding multi-factor authentication.

### 4. Secure Access to AWS Services: 
Be mindful of your app's access to AWS services. Use the least privilege access principle — grant only necessary permissions to your Amplify app.

### 5. Local Testing: 
Amplify provides local mocking and testing capabilities. Use `amplify mock` to test your application locally before deploying changes.

### 6. Keep the CLI Updated: 
AWS regularly updates the Amplify CLI with new features, enhancements, and bug fixes. Keep your CLI updated using `npm install -g @aws-amplify/cli`.

> This guide provides a good starting point for AWS Amplify, but there's a lot more to explore. Make sure to check out the official AWS Amplify documentation for more in-depth information.

