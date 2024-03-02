#incubator 
upstream: [[React Native]]

---

**links**: 

# Expo in React Native Development

## What is Expo?

Expo is an open-source platform for making universal native apps for Android, iOS, and the web with JavaScript and React. It simplifies the process of developing, building, deploying, and quickly iterating on iOS and Android apps by providing a set of tools and services around React Native and native platforms that help developers use advanced features like over-the-air updates, push notifications, and handling assets, without having to touch native code.

Many of Expo's APIs are built on top of the same third-party packages or similar functionality that you would otherwise have to manually integrate into a React Native project. Expo provides a curated set of APIs and components that are pre-integrated and tested to work together seamlessly, simplifying the development process. Here’s how Expo achieves this:

### Simplified Access to Native Features

Expo abstracts away the complexity of directly interacting with native code and managing dependencies. For example, to access the camera or the device's location, instead of installing and linking third-party packages yourself (like `react-native-camera` or `react-native-maps`), you can use Expo's `Camera` or `Location` APIs out of the box. These are designed to offer similar or extended functionalities compared to their standalone counterparts.

### Managed Workflow

In the managed workflow, Expo takes care of the native code and tooling, so you don't have to open Xcode or Android Studio to develop your app. This is particularly beneficial for developers who prefer to focus on writing JavaScript/TypeScript without worrying about native development intricacies. When you use Expo's APIs, you're essentially leveraging Expo's underlying integration with these native functionalities, which might be based on popular third-party packages or custom implementations designed by Expo.

### Custom Native Modules and Bare Workflow

For cases where you need functionality that Expo does not currently offer in the managed workflow, you have the option to "eject" to the bare workflow. This approach allows you to add custom native modules or third-party React Native packages that require linking. In the bare workflow, you can still use many of Expo's convenient features and APIs while gaining full control over the native projects.

### Performance and Optimization

Expo's APIs are optimized for performance and ease of use. By handling the integration and maintenance of these APIs, Expo ensures they are up-to-date and optimized for cross-platform compatibility. This can be particularly advantageous compared to manually integrating various third-party packages, which might have varying levels of maintenance, compatibility, and performance optimizations.

## Why Use Expo Over React Native CLI?

### Simplified Setup and Development Process

- **No Native Code Setup Required:** Expo abstracts away the complexity of setting up native development environments. You don't need to install Xcode or Android Studio to start building your app, making it accessible for developers new to mobile development.
- **Over-the-Air Updates:** Expo allows you to publish updates to your app without going through the app store submission process. This feature enables developers to push fixes and updates quickly and directly to users.
- **Unified Development Experience:** Expo offers a consistent development experience across different platforms, reducing platform-specific bugs and speeding up the development process.

### Access to a Wide Range of APIs

Expo provides a rich set of pre-built APIs and components that cover many use cases, from camera access and location services to push notifications. This reduces the need to use third-party libraries or to write custom native code.

### Expo Go App for Instant Previewing

The Expo Go app allows developers and testers to instantly preview projects on real devices without needing to install any additional software. This speeds up testing and feedback loops.

### Easier Deployment and Publishing

Deploying and publishing apps built with Expo is streamlined, requiring fewer steps compared to traditional React Native apps. Expo handles a lot of the complexity involved in building app binaries for iOS and Android.

## Drawbacks

While Expo offers numerous advantages, especially for beginners and for rapid development, there are some limitations:

- **Limited Control Over Native Code:** For projects that require custom native code beyond what Expo supports, you may need to "eject" to a bare workflow, losing some of the benefits that Expo provides.
- **App Size:** Expo apps might be larger in size compared to vanilla React Native apps because they include the Expo SDK.

## Commonly Used Expo Templates

Expo provides a variety of templates to kickstart project development. These templates cater to different use cases and preferences, including:

- **Blank Template:** A minimal setup for those who want to start from scratch.
- **Tabs Template:** Includes several pre-configured tabs using `react-navigation`, ideal for apps requiring a tab-based navigation system.
- **Bare Workflow Template:** Offers a minimal Expo setup while allowing for custom native code, suitable for developers who need direct access to native capabilities not covered by Expo's managed workflow.

## Conclusion

Expo is a powerful tool for React Native development, offering a streamlined development process, ease of use, and access to a wide range of APIs. It is particularly beneficial for developers looking to quickly prototype and build cross-platform mobile apps without deep diving into native development environments. However, for projects with specific needs that require direct access to native code beyond Expo's capabilities, the traditional React Native CLI approach might be more suitable.