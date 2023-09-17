
###### Upstream: [[Typescript]]
###### Siblings: 
#incubator 

2023-05-21
18:33


### Main Takeaways
- The `?` symbol in TypeScript is used to indicate optional properties in interfaces, types or function parameters.
- I/e: "if you're including the argument, then it should be of this type, but you don't have to include it"
- used in interfaces, function signatures, and keys for objects 

### Why
-   The `?` symbol is used to increase the flexibility of your code, allowing certain properties or parameters to be optionally included.
-   When a property or parameter is marked as optional with the `?`, it means that it can either be of the specified type or it can be `undefined`.
-   This is useful when you want to create an object or call a function, but you don't necessarily have a value for every property or parameter.

### How

- You can use the `?` symbol in an interface or type definition to make a property optional:
```ts
interface Person {   name: string;   age?: number;  // age is optional }
```
In this example, you can create a `Person` object with just a `name` and no `age`.

You can also use the `?` symbol in a function definition to make a parameter optional:
```ts
function greet(name: string, age?: number) {   
	let message = `Hello, ${name}`;   
	if (age !== undefined) {     
		message += `. You are ${age} years old.`;   
	}   
	console.log(message); 
}  
greet("Alice");  
// Prints "Hello, Alice" greet("Bob", 25);  // Prints "Hello, Bob. You are 25 years old."
```
In this example, the `age` parameter is optional, so you can call the `greet` function with either one or two arguments.