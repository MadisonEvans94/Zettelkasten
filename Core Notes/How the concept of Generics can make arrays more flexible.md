
- #incubator 
###### upstream: [[Typescript]], [[Software Development]]

### Origin of Thought:
- Arrays are like the containers that have a label `<T>` which describes the type of items it can hold. So generics are the labels. An explicit use of a generic, i/e: `let output1 = identity<string>("myString");` , is like writting "string" on the metaphorical label 
### Underlying Question: 
- How can the use of generics make arrays more flexible and safe to use in TypeScript?

### Solution/Reasoning: 
-   Generics provide a way to make components work with any data type and not restrict to one data type. So, an array of any type can be processed by a single method, making the code flexible and reusable.
-   Generics can help make arrays more flexible by allowing them to contain any type of data while still providing type safety. This means you can create an array that can contain any type of data, but once you specify the type, TypeScript will enforce that only data of that type is added to the array.
-   Generics also enable you to create functions that can operate on arrays of any type while still preserving the type information.

### Examples (if any): 

```ts
function identity<T>(arg: T[]): T[] { 
	return arg; 
} 
let output1 = identity<string>(["myString1", "myString2"]); 
let output2 = identity<number>([1, 2, 3]);
```

- In this example, the `identity` function is a generic function that can work with arrays of any type. When you call `identity<string>`, TypeScript ensures that the function works correctly with an array of strings. Similarly, when you call `identity<number>`, TypeScript ensures that the function works correctly with an array of numbers. This makes the code more flexible because the same function can work with different types of data, and it also makes the code safer because TypeScript enforces that the data is of the correct type.

### Additional: 
- [[An analogy to be made comparing Generics and Containers]]