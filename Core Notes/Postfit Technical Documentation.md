## Table of Contents

  

---

  

## 1. Introduction

  

### 1.1 Purpose of the Document

  

This technical document serves as a comprehensive guide for the development and implementation of the new GPT-powered chat feature in the PostFit application. The document outlines the architectural decisions, algorithms, and technologies employed to facilitate automated weight loss projection and workout suggestions.

  

### 1.2 Scope

  

- Detailed description of the GPT API and SDK integration.

- Algorithms and mathematical models (specifically **Mifflin**-St Jeor Equations) employed for weight loss projection.

- Data flow and state management for real-time workout suggestions.

- Backend infrastructure modifications and API endpoint definitions.

  

### 1.3 Intended Audience

  

The primary audience for this document consists of backend and frontend developers, data scientists, and QA engineers involved in the PostFit project. A secondary audience includes project managers and technical architects overseeing the development process.

  

---

  

## 2. System Requirements

  

- **Programming languages**: - Dart - Javascript - PHP

> OpenAI provides official libraries and SDKs for many languages. For Dart, check the link [here](https://pub.dev/packages/dart_openai).

- **Development Environment**: You'll need a code editor or integrated development environment (IDE) suitable for your chosen programming language to write and test your integration code.

- **HTTP Client Library**: You will need an HTTP client library for Dart to send HTTP requests to the ChatGPT API. For more details and step by step setup, check section 4.1.

- **API Key**: Obtain an API key from OpenAI for the ChatGPT API. You'll need to include this key as an authorization header in your HTTP requests when calling the API.

- **Hardware requirements**: ChatGPT API requests can be made from cloud servers, local machines, or any environment with internet connectivity. A minimum of 4 GB of RAM is generally adequate for Dart development.

  

---

  

## 3. Architecture Overview

  

### Backend Architecture

  

![](https://lh3.googleusercontent.com/0MQy1P8-R26eJDAvl9bytBh4J7JrkadoXZa0bncAbT_dzmCVhnqFYPVjmNi7srfiWTXq-n4MWTMtcogGjpZ73vUOgT_C440JYw7K-z3g1ghmLorSZEN1UkrzL2UdX_Faok1A_r4ByQryocpu9xo7oxA)

  

> click [here](https://lucid.app/lucidchart/06c2f96f-2bea-46a5-8fb3-7e09da89258b/edit?viewport_loc=-278%2C-87%2C1809%2C887%2C0_0&invitationId=inv_08d10832-d3d8-4eeb-be25-3e444329a501) for source document

  

### Wireframe

  

![](https://lh5.googleusercontent.com/hRYVWn2zUn6w8NAgw01VAOVso94OYa9_qg1eDS1oIS5O56jyU_l1WRumZCPy_cr6qGM0kn-KGcALW9LGETsyYb3tfcEVHlWld1bbP46rIzqmOW9N7XIJbnGV5_NYY0iJWwKsaqdDS9INFKLWYd62rmo)

  

> click [here](https://lh5.googleusercontent.com/hRYVWn2zUn6w8NAgw01VAOVso94OYa9_qg1eDS1oIS5O56jyU_l1WRumZCPy_cr6qGM0kn-KGcALW9LGETsyYb3tfcEVHlWld1bbP46rIzqmOW9N7XIJbnGV5_NYY0iJWwKsaqdDS9INFKLWYd62rmo) for source document

  

---

  

## 4. OpenAI API Setup & Integration

  

To get started, you will need an OpenAI account. Install the OpenAI SDK and instantiate the `OpenAIApi` object with the appropriate configuration. The method of interest is `createChatCompletion`.

  

```js

const openai = new OpenAIApi({ apiKey: "YOUR_API_KEY" });

```

  

For more detailed setup, refer to [OpenAI Documentation](https://platform.openai.com/docs/).

  

### Authentication and Authorization

  

API key-based authentication is used. The API key is set during the instantiation of the `OpenAIApi` object. Ensure the key is securely stored.

  

```js

const configuration = new Configuration({ apiKey: "YOUR_API_KEY" });

```

  

### Rate Limiting and Quotas

  

- **Token Limit**: 4096 tokens max per request.

- **Cost**: $0.002 per 1k tokens.

  

Be aware of the token limitations and costs when making API calls. Utilize token counting libraries like [TIKTOKEN](https://github.com/openai/tiktoken) for Python or [GPT-3-ENCODER](https://www.npmjs.com/package/gpt-3-encoder) for JavaScript to manage token count efficiently.

  

---

  

## 5. New Features

  

### 5.1 Weight Loss Projection Plot

  

#### Calculating Weight Loss Projections

  

The **Mifflin-St Jeor Equation** is employed to estimate the **Resting Metabolic Rate** (RMR) of users. It serves as the foundation for projecting weight loss over a defined period. The equation is:

  

$$

\text{RMR} = 10 \times \text{weight (kg)} + 6.25 \times \text{height (cm)} - 5 \times \text{age (y)} + \text{C}

$$

  

Where $( C = 5 )$ for males and $( C = -161 )$ for females.

  

#### Data Points Used

  

The following metadata from the user serves as input variables:

  

- **Weight** in kg

- **Height** in cm

- **Age** in years

- **Sex**

- Physical activity level (**PAL**)

- **Daily caloric intake**

- **Weight loss goal** in kg

- **Weight loss timeline** in days

  

#### Algorithm for Projection

  

1. **Calculate Initial RMR**: Utilize the Mifflin-St Jeor equation to compute the initial RMR.

2. **Apply Physical Activity Level (PAL)**: Multiply the RMR by the PAL to get the Total Daily Energy Expenditure (TDEE).

3. **Determine Caloric Deficit**: Subtract the daily caloric intake from the TDEE.

4. **Project Weight Loss**: Calculate the weight loss over the defined timeline using the caloric deficit and weight loss goal.

  

> example implementation:

  

```python

def project_weight_loss(weight, height, age, sex, PAL, daily_caloric_intake, timeline_days):

C = 5 if sex == 'male' else -161

RMR = 10 * weight + 6.25 * height - 5 * age + C

TDEE = RMR * PAL

caloric_deficit = TDEE - daily_caloric_intake

```

  

#### Constructing the Prompt for ChatGPT

  

Given the necessary input variables, a prompt for ChatGPT could be constructed as follows:

  

> example prompt:

  

```js

let system_message =

"You are a fitness expert trained to provide weight loss projections and workout suggestions based on the Mifflin-St Jeor equation. Use the given input variables: age, sex, weight, height, PAL, daily_caloric_intake, weight_loss_goal, and weight_loss_timeline.";

  

let user_message = `I am ${age} years old, my sex is ${sex}, my weight is ${weight} kg, my height is ${height} cm, my daily caloric intake is ${daily_caloric_intake} kcal, my PAL is ${PAL}, my weight loss goal is ${weight_loss_goal} kg, and my timeline is ${timeline} days.`;

```

  

The responses from ChatGPT can then be parsed to obtain the necessary data points for plotting.

  

### 5.2 ChatGPT Interactions

  

In this section, we will define a set of built-in prompts that will be used to request workout suggestions from ChatGPT. These prompts serve as the input to the ChatGPT API to initiate conversations and receive workout recommendations.

  

#### 5.2.1 Suggesting Workouts and Steps

  

- **Dynamic prompts**: Make some prompts dynamic by incorporating user-specific information such as the user's name or fitness goals. For example:

  

- _"Can you suggest a workout to help `user_name` achieve `user_goal`?"_

- _"I'm looking to accomplish `user_goal`. Can you suggest steps for that?"_

  

- **Prompt Repository**: Organize the prompts in a structured manner, such as in an array or list, within the Dart code. This makes it easier to iterate through and select prompts for API requests.

  

> Example function call Structure Using Dart (Check section 4.1 for API setup and integration):

  

```dart

import 'package:http/http.dart' as http;

  

Future<String> requestWorkoutSuggestion(String apiKey, String prompt) async {

final apiUrl = 'https://api.openai.com/v1/chat/completions';

final response = await http.post(

Uri.parse(apiUrl),

headers: {

'Authorization': 'Bearer $apiKey',

'Content-Type': 'application/json',

},

body: '{"prompt": "$prompt"}'

)

}

```

  

#### 5.2.2 Caloric Plan Adjustments

  

- **Prompt Definition**: These prompts should be designed to gather relevant information from users and request recommendations from ChatGPT. Example prompts:

- _"I want to achieve `user_goal`. Can you suggest a caloric plan for me?"_

- _"I'm looking to accomplish `user_goal`. What should my daily caloric intake be?"_

- _"Can you help me adjust my caloric plan for `user_goal`?"_

- **Prompt Repository**: Organize the prompts in a structured manner, such as in an array or list, within the Dart code. This makes it easier to iterate through and select prompts for API requests.

  

> Example function call Structure Using Dart (Check section 4.1 for API setup and integration):

  

```dart

import 'package:http/http.dart' as http;

Future<String> requestCaloricPlan(String apiKey, String prompt) async {

try {

final response = await http.post(

Uri.parse(apiUrl),

headers: {

'Authorization': 'Bearer $apiKey',

'Content-Type': 'application/json',

},

body: '{"prompt": "$prompt"}',

);

if (response.statusCode == 200) {

final responseBody = response.body;

return responseBody;

} else {

throw Exception('failed to request plan adjustment')

}

} catch (error) {

throw Exception('error: $error')

}

```

  

## 5.3 Passio: Nutrition-AI Integration

  

### Overview

  

The Passio Nutrition-AI Flutter SDK empowers PostFit with state-of-the-art food recognition and nutrition assistant capabilities. By processing a stream of images, the SDK identifies foods and provides relevant nutritional data.

  

### Key Features:

  

- **Real-time Food Recognition**: As users hover their camera over food items, the SDK identifies them without storing any photos or videos.

- **Comprehensive Outputs**: The SDK provides:

- Recognized food names (e.g. banana, hamburger).

- Alternatives for recognized foods (e.g., soy milk as an alternative to milk).

- Barcodes and text from food packages.

- Nutrition facts via Passio's reader.

- Food weight and volume for select foods.

  

### Data Sources:

  

- The SDK incorporates data from [Open Food Facts](https://en.openfoodfacts.org/). Foods sourced from this database will have the `isOpenFood: Bool` attribute. Developers must adhere to Open Food Facts' [license agreement](https://opendatacommons.org/licenses/odbl/1-0) and [terms of use](https://world.openfoodfacts.org/terms-of-use).

  

### Setup and Configuration:

  

1. **SDK Access**: Developers need to sign up at [Passio's Nutrition-AI](https://www.passio.ai/nutrition-ai) to obtain a valid SDK key.

2. **Device Requirements**:

- Android: SDK 26+

- iOS: 13.0+

- Camera access is mandatory.

3. **Initialization**:

- Import the SDK: `import 'package:nutrition_ai/nutrition_ai_sdk.dart';`

- Configure with the obtained key and handle the SDK's status, ensuring it's ready for food detection.

  

### Usage:

  

1. **Camera Authorization**: Ensure permission for camera usage is granted.

2. **Start Food Detection**: Implement the `startFoodDetection()` method and register a `FoodRecognitionListener` for real-time results.

3. **Stop Detection**: It's crucial to stop the detection process once it's no longer needed, typically during widget disposal.

  

### Data Models:

  

The SDK offers a variety of classes to represent different data structures, including:

  

- `FoodCandidates`: Represents SDK's detection results.

- `PassioFoodItemData`: Contains nutritional info for a food item.

- `PassioFoodRecipe`: Holds nutritional info for food classified as a recipe.

  

Developers can explore further classes, extensions, and enums provided by the SDK for more comprehensive integration.

  

> click [here](https://passio.gitbook.io/nutrition-ai/) for Passio Nutrition SDK documentation

>

> click [here](https://pub.dev/documentation/nutrition_ai/latest/nutrition_ai/nutrition_ai-library.html) for details on the api utility classes in Nutrition API

>

> click [here](https://pub.dev/documentation/nutrition_ai/latest/) for _flutter specific_ Passio SDK Documentation

  

---

  

## 6. Data Models

  

![](https://lh6.googleusercontent.com/MYC318uQOduiDdBBbAhh99KweuirhV80LHqQOHOV6xPYRX5cVHf3kDABedUgxGZ_aKdcv7uKAIrJev7-ynvxnZwrNPMfjtkwXA-gwR8EM60ULFRNWrLJQ7aDsujOKGYyLM1MggvoO3VC-VvMZm2T7w0)

  

> click [here](<**[https://lucid.app/lucidchart/35c9ffd2-1e09-48f6-aa94-a25f8049e16c/edit?viewport_loc=374%2C-82%2C1867%2C961%2C0_0&invitationId=inv_e08c2bfa-d514-4b22-af73-e4031732e77e](https://lucid.app/lucidchart/35c9ffd2-1e09-48f6-aa94-a25f8049e16c/edit?viewport_loc=374%2C-82%2C1867%2C961%2C0_0&invitationId=inv_e08c2bfa-d514-4b22-af73-e4031732e77e)**>) for source document

  

---

  

## 7. Integration with Existing Features

  

This section outlines how nutrition tracking aligns with and complements existing features like workout and diet plans.

  

### 7.1 Nutrition Tracking and Diet Plans

  

Nutrition tracking seamlessly integrates with the diet planning feature. Users can now not only set dietary goals but also monitor their progress effectively. Here's how it works:

  

Dietary Goals: Users can define their dietary goals, such as calorie intake, and specific dietary preferences (e.g., vegetarian, keto, etc.).

  

Food Logging: Nutrition tracking feature allows users to log the foods they consume throughout the day.

  

Real-time Nutrition Data: The integration with Passio provides real-time nutritional information for logged foods, including calories. Users can instantly see how their dietary choices align with their goals.

  

### 7.2 Nutrition Tracking and Workout Plans

  

Incorporating nutrition tracking alongside workout plans creates a holistic approach to fitness:

  

Workout Performance: Users can now correlate their nutrition choices with their workout performance. For instance, they can assess how their energy levels and recovery are influenced by their dietary habits.

  

Caloric Needs: It can calculate users' daily caloric needs based on their workout plans and goals. This ensures that users are fueling their workouts adequately and optimizing their training.

  

Weight Management: The integration of nutrition tracking contributes to accurate weight management. By monitoring both diet and exercise, users can achieve a more balanced approach to weight loss or muscle gain.

  

### 7.3 Comprehensive Wellness

  

Ultimately, the integration of nutrition tracking into existing features aims to provide users with a comprehensive view of their wellness journey. By combining data from workouts, diet plans, and nutrition tracking, users can make informed decisions to achieve their fitness and health goals.

  

---

  

## 8. Milestones and High-Level Timeline

  

#### Milestone 1: Project Kickoff and Planning (1-2 weeks)

  

##### Week 1:

  

Project kickoff and team orientation.

Review and approval of the project plan.

Environment setup, including Dart development environment and API access.

  

##### Week 2:

  

Detailed design discussions for UI, prompts, and function calls.

Finalize prompts for new features.

Begin initial design of user interface components.

  

#### Milestone 2: Development and Integration (4-6 weeks)

  

##### Weeks 3-4:

  

UI design and development for new AI fitness plan feature.

Implementation of function calls for ChatGPT API integration.

  

##### Weeks 5-6:

  

Implement logic for parsing and displaying new feature.

Begin testing and debugging of the integrated feature.

  

#### Milestone 3: Testing and User Acceptance (2-3 weeks)

  

##### Weeks 7-9:

  

Unit testing of function calls, response handling, and UI components.

Integration testing to ensure feature functionality and performance.

Begin planning for user acceptance testing (UAT).

  

#### Milestone 4: Documentation and Final Testing (1-2 weeks)

  

##### Weeks 10-11:

  

Complete user documentation for the AI Fitness Plan feature.

Conduct UAT with a group of real users.

Gather feedback and make necessary adjustments based on user input.

  

#### Milestone 5: Deployment and Monitoring (1 week)

  

##### Week 12:

  

Release the app update, including the new feature.

Monitor app performance and gather user feedback after the update.

  

#### Milestone 6: Post-Release Support and Iterations (Ongoing)

  

##### Ongoing:

  

Provide ongoing support and maintenance for the feature, addressing any bugs or issues.

Continuously analyze user feedback and data to make improvements and refinements to the feature.

  

##### Release Target:

  

Target Release Date: Approximately 12-14 weeks from the project kickoff.

  

---

  

## 9. User Interface Components

  

- New Widgets and Views

- Weight Loss Projection Plot

- Schedule Workout Page

  

---

  

## 10. Error Handling and Logging

  

- Types of errors

- Logging mechanisms

- User feedback

  

---

  

## 11. Testing

  

- Unit Testing

- Integration Testing

- End-to-End Testing

  

---

  

## 12. Deployment

  

- Build process

- Deployment steps

- Rollback procedures

  

---

  

## 13. Appendix

  

- Code snippets

- Additional resources

  

---

  

## 14. Glossary

  

- Definitions of technical terms

  

---

  

## 15. References

  

- ChatGPT API [documentation](https://docs.google.com/document/d/1-eVyf7tbXIFq3Zw6noxqrplOEJ4pxFWPKj5a4G61qwk/edit?usp=sharing)

- Dart SDK [documentation](https://pub.dev/packages/dart_openai)

- Passio AI sdk [documentation](https://pub.dev/packages/nutrition_ai)

- Passio Nutrition [documentation](https://www.passio.ai/nutrition-ai)