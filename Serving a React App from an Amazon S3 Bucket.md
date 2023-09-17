#seed 
upstream: [[React]], [[AWS]]

---

**video links**: 
https://www.youtube.com/watch?v=SHN48wTEQ5I&ab_channel=DevGuyAhnaf

---


Sure, here's a nicely structured Markdown version of the process:

---

This guide outlines the steps to serve a React application, built using create-react-app, from an Amazon S3 bucket.

## Prerequisites
- You have a created **React** application.
- You have an AWS account and an **S3** bucket ready.

## Step 1: Build Your React Application

Before you can upload your React app to S3, you need to create a production build of your app. This can be done by running the following command in your project's root directory:

```bash
npm run build
```

This command generates a `build` directory containing a production-ready, optimized version of your app.

you should get this response on a successful build: 

```bash 
The project was built assuming it is hosted at /.
You can control this with the homepage field in your package.json.

The build folder is ready to be deployed.
You may serve it with a static server:

  npm install -g serve
  serve -s build

Find out more about deployment here:
```

You should upload everything inside the `build` folder, not just the `static` directory. 

This is because your `index.html` file, along with the other files like `asset-manifest.json`, `manifest.json` and `robots.txt`, are also essential for your React application to run properly.

The `index.html` file is the entry point of your application and it references JavaScript and CSS files that are inside the `static` directory. `asset-manifest.json` and `manifest.json` are used by the service worker for caching and updating your application, while `robots.txt` is used by search engines for crawling and indexing your application.

When you're using the AWS CLI or the AWS Management Console, it will recursively upload all directories and files under the `build` directory. So, if there are any directories inside the `build` directory (like `static` and `Molika`), it will upload those directories along with all of their contents.

> So, to make it clear, upload all the contents of the `build` directory, including all files and directories.

## Step 2: Upload the Build to Your S3 Bucket

After building your application, the next step is to upload it to your **S3** bucket. You can do this either manually via the AWS Management Console or programmatically using the AWS CLI.

### Option 2.1: Uploading via the AWS Management Console
Go to the S3 section in the AWS Management Console, navigate to your bucket, and manually upload the contents of your build directory.

> Important: Make sure to upload the contents of the build folder, not the folder itself.

### Option 2.2: Uploading via the AWS CLI
If you have AWS CLI installed and configured on your machine, you can use the following command to upload your build folder to your bucket:

```bash
aws s3 sync build/ s3://mybucket
```

Replace `mybucket` with the name of your S3 bucket.

## Step 3: Configure S3 for Web Hosting

Next, enable static website hosting in your S3 bucket. This can be done by:

1. Going to the Properties of your bucket.
2. Turning on the "Static website hosting" option.
3. Setting the Index Document field to `index.html`.
4. Optionally, setting the Error Document field to `404.html` if you have a custom error page.

## Step 4: Set the Bucket Policy

You must modify the bucket policy to allow public read access for your website to be accessible publicly. This can be done by:

1. Going to the "Permissions" tab in your bucket settings.
2. Adding the following policy in the Bucket Policy editor:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "PublicReadGetObject",
            "Effect": "Allow",
            "Principal": "*",
            "Action": "s3:GetObject",
            "Resource": "arn:aws:s3:::mybucket/*"
        }
    ]
}
```

Don't forget to replace `mybucket` with the name of your bucket.

## Step 5: Access Your Website

After completing all these steps, you can access your website using the endpoint provided in the Static website hosting card in your bucket properties. The URL will be in the following format:

```
http://mybucket.s3-website-us-east-1.amazonaws.com
```

## Note on HTTPS

Hosting your website directly on S3 does not provide HTTPS encryption. To enable HTTPS, consider using a service like AWS CloudFront or another suitable method.

*see [[adding a custom domain]] for more details on custom domains*

---



