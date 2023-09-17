
## Introduction 

The ChatGPT API allows you to integrate the ChatGPT language model into your own applications, products, or services. It's useful for automating various kinds of conversational tasks.

---
## Getting Started 

To get started with the chatgpt api, you will first need an openAI account. The service that will be used for chatgpt is called **Chat Completion**. To integrate into a codebase, use the openAI sdk found [here](https://platform.openai.com/docs/libraries/community-libraries)

---
## OpenAI Classes

### The `openai` Object

When you import `OpenAIApi` from the OpenAI SDK and instantiate it with `new OpenAIApi(configuration)`, you're creating an object that has several methods for interacting with different OpenAI APIs. The one we're focusing on is `createChatCompletion`.


```js
const openai = new OpenAIApi(configuration);
```

- **Method: `createChatCompletion`**
    
    - **Purpose**: To initiate a chat-based language task with GPT-3.5 Turbo or its variants.
        
    - **Parameters**: An object containing `model` and `messages`.
        
        - `model`: Specifies the language model to use (e.g., `'gpt-3.5-turbo'`).
            
        - `messages`: An array of message objects, each with a `role` ("system", "user", "assistant") and `content` (the text content of the message).
            
    - **Return Value**: An asynchronous Promise that resolves to a `chatGPTResponse` object.
        

### The `chatGPTResponse` Object

The `chatGPTResponse` object encapsulates the response data from the ChatGPT API call. This object is structured with nested attributes, and here we'll dissect the attributes used in this specific line of code: `chatGPTResponse.data.choices[0].message.content`.

- **`data`**: This attribute holds the actual response from the API. It's an object that contains several sub-attributes.
    
    - **`choices`**: An array of "choice" objects. Since GPT-3.5 Turbo is a single-choice model, you'll generally only see one choice, accessed by `choices[0]`.
        
        - **`message`**: An object that contains information about the assistant's generated message.
            
            - **`content`**: This is the final text generated by the model. It is what you usually display to the end-user.

>Here's a JSON sample for visual representation:

```json
{
  "data": {
    "choices": [
      {
        "message": {
          "role": "assistant",
          "content": "The Dodgers won the World Series in 2020."
        },
        // ... other attributes
      }
    ],
    // ... other attributes
  }
}
```

---
## Message Fields 

The ChatGPT API request uses a `messages` array that contains objects with two fields: `role` and `content`.

In the context of using the OpenAI API for ChatGPT, the `messages` array can indeed include messages from three roles: `system`, `user`, and `assistant`.

1. **System**: This is where you put your primer or context that helps guide the model's behavior. The `system` role helps in priming the assistant to function in a specific way.

2. **User**: This is where the user's question or prompt is placed. The assistant uses the context provided by the `system` role and the `user` role to generate a response.

3. **Assistant**: This field is typically used for back-and-forths with the model. It's the field where you put the model's prior responses when you're having a multi-turn conversation. When you initially send a message to the model, you typically won't include this field yourself. Instead, it gets filled in with the model's responses as the conversation progresses.

In single-turn tasks, you'll often see just the `system` and `user` roles being used. The `assistant` role becomes important in multi-turn conversations where the model's prior responses need to be considered for context.

>example: `messages` array that will be formed on client and sent to backend 

```js

let messages = [
	{role: "system", content: "primer statement goes here..."}, 
	{role: "user", content: "prompt/request goes here..."}, 
	{role: "assistant", content: "ChatGPT response here..."}
]
```

---
## Configuration

Let's outline how the setup/configuration will look like on the server side. `model` parameter allows you to specify which language model to use. As of now, `gpt-3.5-turbo` is the most advanced.

>example: server side implementation receiving `messages` array from client

```js
import {Configuration, OpenAIApi} from 'openai'; 

const configuration = new Configuration({
	apiKey: 'YOUR_API_KEY', 
})

const openai = new OpenAIApi(configuration); 

export const POST = async ({ request, response }) => {
  const { messages } = await request.json();
  const chatGPTResponse = await openai.createChatCompletion({
    model: 'gpt-3.5-turbo',
    messages,
  });
  response.status = 200;
  response.body = { reply: chatGPTResponse.data.choices[0].message.content };
};

```

### Advanced Configuration

Apart from the basic configuration that includes specifying the API key and language model, you can also include advanced parameters to fine-tune the model's behavior.

- **`temperature`**: Controls the randomness of the output. A lower value (e.g., 0.2) makes the output more deterministic, while a higher value (e.g., 0.8) makes it more random.

- **`max_tokens`**: Limits the length of the generated response. Be cautious when setting this value to ensure you don't cut off meaningful output.

```js
const chatGPTResponse = await openai.createChatCompletion({
    model: 'gpt-3.5-turbo',
    messages,
    temperature: 0.5,
    max_tokens: 100,
});
```
---
## Using Dynamic Values 

In practice, we'll want to use dynamic values in order to accomplish a programatic approach. Here's the same example from above, but using dynamic values 

>example of singe turn task

```js
let messages = [
{role: "system", content: "You are a fitness trainer. Your prime function is to return numerical data that will be used for fitness suggestions and weight loss forecasts based on input received from the user. the input variables will be as follows: age(years), sex, weight(kg), height(cm), weight_loss_goal(kg), daily_caloric_intake, weight_loss_timeline(days), fitness level(1-10). given this information, you are to suggest a detailed weekly fitness plan"}, 
{role: "user", content: `I am ${age} years old. My sex is ${sex} . My weight is ${weight}. My height is ${height}. My daily caloric intake is ${daily_caloric_intake}. My fitness level is ${fitness_level}. My current weight loss goal is ${weight_loss_goal}. My timeline is ${timeline}.`}
]
```

---
## Limitations 

- The chatgpt api has a max token limitation of **4096** tokens. A token in this case is a word or partial word
- The price for the model is **$0.002 / 1k tokens**

>Note: one useful tool for managing the count of tokens used per request is a library called [TIKTOKEN](https://github.com/openai/tiktoken) for **python** or [GPT-3-ENCODER](https://www.npmjs.com/package/gpt-3-encoder) for **javascript** 