#seed #incubator 
###### upstream: [[Software Development]], [[Typescript]]

## TLDR: 


### Origin of Thought:
- Improving understanding by thinking of use cases 

### Underlying Question: 
- When should you use Generics and what are some best practice use cases? 

### Solution/Reasoning: 
Generics are a tool in TypeScript (and many other statically typed languages) that allow you to write flexible, reusable code while still maintaining type safety. Here are some situations where using generics can be beneficial:

**1. Creating Reusable Utility Functions:**

Generics are great when you want to create utility functions that can work with different types of data while still providing type safety. For instance, consider a function that returns the first element in an array. Without generics, you might write something like this:

```ts
function getFirstElement(arr: any[]): any {   
	return arr[0]; 
}
```

This works, but it's not type safe: you lose information about what type of element is in the array. With generics, you can do this:


```ts
function getFirstElement<T>(arr: T[]): T {   
	return arr[0]; 
}
```

Here, `T` is a placeholder for any type. This means `getFirstElement` can be used with an array of any type, and it will return a value of that type.

**2. Creating Type-Safe Data Structures:**

If you're creating a class for a data structure like a Linked List or a Binary Tree, you'll likely want to use generics. This allows your data structure to contain any type of data, while still maintaining type safety. For example:


```ts
class LinkedListNode<T> {   
	constructor(public data: T, public next: LinkedListNode<T> | null = null) {} 
}
```

**3. Wrapping Other Types:**

Sometimes you might want to create a type that "wraps" another type, like a `Promise<T>` in TypeScript, which represents a value that may not be available yet. Here, `T` can be any type, so a `Promise<string>` is a promise that will eventually produce a string, a `Promise<number>` will produce a number, and so on.

**Best Practices:**

-   Use meaningful names for your generic type variables. While it's common to use `T` as a generic type variable, sometimes it's helpful to use a more descriptive name, especially when your function or class takes multiple generic types.
    
-   Avoid using generics when they're not needed. Sometimes a simple type will do just fine, and using generics would only complicate things.
    
-   Don't forget that generics can extend types. For example, `T extends string` means that `T` can be any subtype of string. This can be useful for constraining what types can be used with your generic function or class.
    
-   Remember that TypeScript's type inference often works with generics. If TypeScript can figure out what type `T` should be based on the arguments you're passing to a function, you often don't need to explicitly specify `T`.
    

Overall, generics are a powerful tool for creating flexible, reusable, type-safe code. Use them when they make your code clearer and safer, but don't overuse them to the point where they make your code complex and hard to understand.

### Examples (if any): 

