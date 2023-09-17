#evergreen1
###### upstream: [[Typescript]] , [[Software Development]]

### Origin of Thought:
- While learning typescript, I learned some use cases for Tuples, which is a new datastructure that Javascript doesn't have by default 
- Tuple seems to have similarities (in terms of use cases) that are found in Objects, and I'm not sure what Tuples solve that Objects don't 

### Underlying Question: 
- What do Tuples solve that normal objects can't? 

### Solution/Reasoning: 
- The only difference between the two is how you access the info
- Both are built off of a javascript object under the hood, except for tuples, position matters... Regardless, both have [[Constant Time Lookup]]
- Because index matters in tuples, they are a good choice when we need to iterate. When you create a tuple, you know that the first element will always be the first element, the second will always be the second, and so on. This makes it straightforward to iterate over a tuple in order
- Objects are more efficient for lookups when the keys are not known in advance or when there are a large number of keys, since object property access is a constant-time operation. If you're dealing with larger, unordered collections of data with arbitrary keys, objects would generally be more efficient.

### Examples (if any): 


While both tuples and objects are versatile and can solve a wide range of problems, there are certain scenarios where the use of a tuple might be more appropriate or convenient than an object.

Consider the following scenario: you need to write a function that returns two values, for instance, a string and a number. With tuples, you can return both values in one neat package:


```ts
function getPersonInfo(): [string, number] {     
	let name = 'Alice';     
	let age = 25;     
	return [name, age];  
}
// returns a tuple }  let info = getPersonInfo(); console.log(`Name: ${info[0]}, Age: ${info[1]}`);
```

In the above code, the function `getPersonInfo` returns a tuple with a `string` and a `number`. Using an object to achieve this would be more verbose, and accessing the returned values would require knowing the property names:


```ts
function getPersonInfo(): {name: string, age: number} {     
	let name = 'Alice';     
	let age = 25;     
	return {name, age}; 
}
	// returns an object }  let info = getPersonInfo(); console.log(`Name: ${info.name}, Age: ${info.age}`);
```

Note that both solutions are correct, and whether to use a tuple or an object often comes down to the specific needs of your program and coding style. However, in this scenario, if you only need to return a fixed, ordered set of values and don't need additional methods or properties, a tuple can provide a more concise solution.

### Additional Questions (if any): 

- [ ] 
