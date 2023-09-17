#seed 
upstream:

---


AWS Amplify makes handling authentication in your React app a breeze. In this guide, we'll cover how to configure Amplify in an existing React project, and how to connect it to a User Pool and Identity Pool created through the AWS Management Console.

## Introduction to Amplify and Authentication

AWS Amplify provides a high-level interface to [[AWS Cognito]] - AWS's service for managing user identities and authentication workflows. Cognito is split into two parts: **User Pools** and **Identity Pools**. User Pools handle user management and authentication, while Identity Pools handle authorization of authenticated users to use AWS services.

## Setting up Amplify in a React Project

First, install AWS Amplify:

```bash
npm install aws-amplify
```

Next, initialize Amplify in your project. In your `index.js` (or similar entry file), import Amplify and configure it:

```javascript
import Amplify from 'aws-amplify';
import awsconfig from './aws-exports';

Amplify.configure(awsconfig);
```

The `aws-exports.js` file is created by the Amplify CLI and contains the configuration for your backend resources. If you haven't used the CLI to create resources, you'll need to create this manually.

## Connecting Amplify to a User Pool

To connect Amplify to a User Pool, you need to provide the User Pool's ID and App Client ID to the Amplify configuration. These can be found in the AWS Management Console, under the Cognito service. 

Your `aws-exports.js` should look something like this:

```javascript
const awsconfig = {
  Auth: {
    region: 'your-region',
    userPoolId: 'your-user-pool-id',
    userPoolWebClientId: 'your-user-pool-web-client-id',
  },
};

export default awsconfig;
```

## Connecting Amplify to an Identity Pool

To connect Amplify to an Identity Pool, you need to provide the Identity Pool's ID. This can also be found in the AWS Management Console, under the Cognito service. Add this to the Amplify configuration in your `aws-exports.js`:

```javascript
const awsconfig = {
  Auth: {
    region: 'your-region',
    userPoolId: 'your-user-pool-id',
    userPoolWebClientId: 'your-user-pool-web-client-id',
    identityPoolId: 'your-identity-pool-id',
  },
};

export default awsconfig;
```

## Navigating to Hosted UI for Sign In

To initiate the sign in flow, you can use the `Auth.federatedSignIn()` method from Amplify, which will take the user to the Hosted UI:

```javascript
import { Auth } from 'aws-amplify';

const signIn = async () => {
  await Auth.federatedSignIn();
};
```

You can call this function from an onClick handler in your sign in button:

```jsx
<button onClick={signIn}>Sign In</button>
```

After successfully signing in, the user will be redirected back to your app.

## Protecting Routes in a React Application

To protect routes in your application, you'll need to check whether the user is authenticated. Amplify provides the `Auth.currentAuthenticatedUser()` method for this:

```javascript
import { Auth } from 'aws-amplify';

const checkUser = async () => {
  try {
    const user = await Auth.currentAuthenticatedUser();
    console.log('User is signed in:', user);
  } catch (error) {
    console.log('User is not signed in');
  }
};

checkUser();
```

In a React Router-based app, you can use this to conditionally render routes:

```jsx
<Route
  path="/protected"
  render={() =>
    isAuthenticated ? (
      <ProtectedRoute />
    ) : (
      <Redirect to="/login" />
    )
  }
/>
```

Make sure to call `checkUser()` when your app starts, and whenever the user's authentication state may have changed (for example, after they are redirected back from the hosted UI).

This guide gives you a solid starting point for handling authentication with AWS Amplify in a React app. Always refer back to the Amplify and Cognito documentation for more complex scenarios and capabilities.

- [x] if I didn't want to use amplify for the auth stuff, how would I do it?
You can definitely use AWS Cognito directly without AWS Amplify. You would have to interact with AWS Cognito using its SDK, the AWS SDK for JavaScript in the Browser, which you can install using npm.

>see this [video](https://www.youtube.com/watch?v=8WZmIdXZe3Q&ab_channel=EvanDoesTech) series for step by step walkthrough

```bash
npm install aws-sdk
```

Here are the steps:

1. **Import the necessary AWS SDKs and configure them**:

At the top of your JavaScript or React file, import the AWS SDK:

```javascript
import AWS from 'aws-sdk';
```

And then, configure the region:

```javascript
AWS.config.region = 'us-west-2'; // Your region
```

To use Cognito, import `CognitoIdentityServiceProvider`:

```javascript
const CognitoIdentityServiceProvider = AWS.CognitoIdentityServiceProvider;
```

Then, instantiate a CognitoIdentityServiceProvider:

```javascript
const cognitoIdentityServiceProvider = new CognitoIdentityServiceProvider();
```

2. **Sign In User**:

Here's an example function that signs a user in:

```javascript
const signIn = (username, password) => {
	const params = {
		AuthFlow: 'USER_PASSWORD_AUTH', 
		ClientId: 'your-client-id', 
		AuthParameters: {
			USERNAME: username,
			PASSWORD: password
		}
	};

	cognitoIdentityServiceProvider.initiateAuth(params, (err, data) => {
		if (err) console.error(err);
		else console.log(data);
	});
};
```

3. **Check User Authentication**:

Once signed in, AWS Cognito returns tokens that you can use to check if the user is authenticated. The tokens are returned in the previous `signIn` function. Here's an example of how you can use them:

```javascript
const isAuthenticated = (accessToken) => {
	const params = {
		AccessToken: accessToken
	};

	cognitoIdentityServiceProvider.getUser(params, (err, data) => {
		if (err) console.error(err);
		else console.log(data);
	});
};
```

4. **Protecting Routes**:

In your routing component (e.g., React Router), you can use the `isAuthenticated` function to conditionally render components based on authentication status. Note that this is an asynchronous operation, and you may want to manage this state using hooks or state management libraries:

```jsx
<Route
	path="/protected"
	render={() =>
		isAuthenticated ? (
			<ProtectedComponent />
		) : (
			<Redirect to="/login" />
		)
	}
/>
```

Keep in mind that this way of implementing AWS Cognito requires managing tokens manually and dealing with raw responses from the AWS SDK. AWS Amplify simplifies these aspects and provides additional features, but this approach gives you more flexibility if you need it.