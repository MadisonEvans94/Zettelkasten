#seed 
###### upstream: [[Typescript]]
###### siblings: [[signatures are to interfaces as anonymous functions are to defined functions]]
### Origin of Thought:
- While learning typescript, I came across the concept of a function signature, and I didn't know what it was, but the description sounded like an interface, so I wanted to look more into what a signature is and how it's different than an interface

### Underlying Question: 
- What is a signature and how is it different than an interface? 

### Solution/Reasoning: 
- function signatures and function interfaces solve the same problem: creating a structural contract for functions to follow
- Interfaces can be more beneficial in the following scenarios:

1.  **Reusable Function Types:** If you have a function type that's used in multiple places, it can be helpful to define an interface for that type. This lets you name the type and use that name instead of repeating the entire function signature every time.
    
    typescriptCopy code
    
```ts
	interface StringToStringFunc {   
		(input: string): string; 
	}  
// Now you can use StringToStringFunc as a type anywhere in your code. let uppercase: StringToStringFunc; let lowercase: StringToStringFunc;
```
    
2.  **Complex Types:** If you're working with a complex type that includes multiple function signatures, interfaces can help make your code cleaner and easier to understand.
    
    typescriptCopy code
    
    `interface StringManipulation {   uppercase(input: string): string;   lowercase(input: string): string;   capitalize(input: string): string; }`
    
3.  **Contracts for Classes or Objects:** If you want to define a contract that certain classes or objects must adhere to, interfaces are the way to go. Any class that implements an interface must provide an implementation for each function defined in the interface.
    
    typescriptCopy code
    
    ``interface Greeting {   greet(name: string): string; }  class EnglishGreeting implements Greeting {   greet(name: string) {     return `Hello, ${name}`;   } }  class FrenchGreeting implements Greeting {   greet(name: string) {     return `Bonjour, ${name}`;   } }``

### Examples (if any): 

