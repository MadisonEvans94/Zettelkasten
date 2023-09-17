#incubator 
###### upstream: [[Typescript]], [[Software Development]]

### Origin of Thought:
- Interfaces are a new addition to Javascript via Typescript and I want to familiarize myself with what makes them different than classes 

### Underlying Question: 
- What differences are there between classes and interfaces, and what does one solve that the other doesn't? 


### Solution/Reasoning: 
Interfaces are like **contracts** that your code must follow in order to compile correctly, whereas classes are more like blocks of code that hold the actual implementation detail for object creation and functions within their scope

Classes: 
- They have the ability to **create** instances 
- They carry actual implementation detail (methods)
- They exist in compile time and runtime 
- Classes provide access modifiers (in Typescript) 

Interfaces: 
- They do not exist at runtime, only compile time. This is because [[When Typescript Compiles to Javascript]], it only uses the interface as an instruction on how to type check during compile time
- interfaces provide a more flexible way to define [[function types and index signatures]]

### Additional Notes: 

1.  **Inheritance and Extending**: Both classes and interfaces can be extended in TypeScript, allowing for a hierarchical structure. However, classes can only extend a single class (single inheritance), whereas an interface can extend multiple interfaces.
    
2.  **Implementing Interfaces**: Classes in TypeScript can implement one or more interfaces using the `implements` keyword. This is a powerful feature that ensures the class adheres to the contract specified by the interface(s).
    
3.  **Optional Properties**: Interfaces allow for optional properties, which can provide flexibility when defining object shapes. Classes don't have optional properties in the same way, but they can achieve a similar effect by assigning a default value or `null` to a class property.
    
4.  **Access Modifiers**: Classes in TypeScript support access modifiers (`public`, `private`, and `protected`), which control the visibility of class members. Interfaces, on the other hand, do not have access modifiers because they don't include an implementation.
    
5.  **Read-Only Properties**: Interfaces can have read-only properties, which are properties that can be assigned a value when the object is created, but can't be changed after that. This can be useful when you want to create an object with a property that should not be modified.

### Examples (if any): 

```ts
interface Animal {
  readonly name: string;
  sound?: string;  // Optional property
}

class Dog implements Animal {
  readonly name: string;
  sound = "woof";

  constructor(name: string) {
    this.name = name;
  }

  makeSound() {
    console.log(this.sound);
  }
}

let dog = new Dog("Rover");
dog.makeSound();  // Prints "woof"

```

See [[The "?" Symbol in Typescript]] for more