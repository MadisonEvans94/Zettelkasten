
###### Upstream: [[Typescript]]
###### Siblings: 
#seed 

2023-05-20
13:22


### Details: 
In TypeScript, type casting (also referred to as type assertion) is a way to inform the TypeScript compiler that you know more about the type of a variable than TypeScript can infer on its own.

Type casting in TypeScript is performed with either the `as` keyword or angle brackets `<>` syntax. However, the `<>` syntax can be confused with JSX syntax, so the `as` keyword is generally preferred in TypeScript code.

Here's an example of type casting in TypeScript:
```ts
let someValue: unknown = "this is a string"; 
let strLength: number = (someValue as string).length;
```

In the above example, `someValue` is of type `unknown` (see [[Difference between type unknown and type any]] for more). TypeScript won't let you access the `.length` property of `someValue` because it doesn't know if `someValue` is a string. By using type casting (`as string`), you're informing TypeScript that `someValue` is indeed a string, and accessing the `length` property is safe.

### Main Takeaways
1.  **Type casting does not change the actual type or value of a variable.** It is only a way to inform TypeScript about the type of the variable.
    
2.  **Type casting does not perform any special checking or restructuring of data.** It's up to the developer to use it correctly.
    
3.  **Type casting doesn't have any runtime impact.** TypeScript's type system exists only at compile time. Type casting, like all other TypeScript types, is removed during the transpilation process to JavaScript.
### Why
Type casting is used in situations where you know more about the type of an object than TypeScript does. It's a way to give additional information to the TypeScript compiler that can't be inferred from the code.

For example, if you are using `document.getElementById`, TypeScript can't know the specific type of the element being returned (it could be any subtype of `HTMLElement`). If you know it's going to be an `HTMLInputElement`, you could use type casting to inform TypeScript about this.

### How

Here's an example with a DOM element:

```ts
const element = document.getElementById('my-input') as HTMLInputElement; 
console.log(element.value); 
// TypeScript now knows that `element` has a `value` property.
```

In this example, you're telling TypeScript to treat `element` as an `HTMLInputElement`. This allows you to access properties and methods like `value` that exist on `HTMLInputElement` but not on the more general `HTMLElement` type.

However, remember that you need to be careful while using type casting. Incorrect type casting can lead to errors because you're essentially telling TypeScript to bypass its usual type checks.

