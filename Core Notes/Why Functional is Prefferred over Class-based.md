#incubator 
###### upstream: [[React]]

### Origin of Thought:
- In React, it seems like functional took popularity over class based. By understanding this, I may better understand the general ecosystem for this flavor of frontend development

### Underlying Question: 
- In modern React JS and libraries like it, why does it seem like there's been a general shift towards preferring functional based components as opposed to class based? Are there benefits or is this trend arbitrary? 

### Solution/Reasoning: 
The shift from class-based components to functional components in React (and similar libraries/frameworks) is not arbitrary. It is driven by a combination of technical and developer experience reasons.

1.  **Simplicity:** Functional components are just functions. This makes them easier to read and understand, especially for developers coming from a functional programming background.
    
2.  **[[Hooks]]:** With the introduction of Hooks in React 16.8, you can now use state and other React features without writing a class. This has greatly increased the usability of functional components. It is easier and cleaner to use hooks with functional components than to manage lifecycle methods in class components.
    
3.  **Reduced Boilerplate:** Functional components have less boilerplate. You don't need to write constructors, bind event handlers, or remember to call `super(props)`.
    
4.  **Performance:** Functional components can be less expensive in terms of memory and performance, although the differences are often minor and don't affect smaller applications. React can also optimize functional components more in the future.
    
5.  **Improved Testing:** Functional components are generally easier to test and debug because they are pure functions (given the same props, they will always produce the same output).
    
6.  **Improved Hot Reloading:** When you change class components, the state is often reset due to how the hot reloading works. With Hooks and functional components, it is easier to keep the state while changing the component.
    
7.  **Future of React:** The React team has mentioned that they will continue supporting class components, but they also stated that they see Hooks as the future. They have been encouraging developers to start trying Hooks in new components.
    
8.  **Community Trend:** The trend in the community is moving towards functional components. More and more examples, tutorials, and open-source projects are using functional components.

Summary: Functional components are less complex to implement, and whenever you need to use somehting from React, you can just *hook* it into your functional component 

### Examples (if any): 

### Additional Thoughts: 

- [[What is a good comparison or analogy for Hooks?]]