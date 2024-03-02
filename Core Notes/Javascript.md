#seed 
upstream: [[Software Development]]

---

**links:**

**siblings:** [[Lodash]]

---

## Data Structures

JavaScript, as a flexible and dynamic language, supports various data structures that enable developers to handle and manipulate data efficiently. Understanding these structures is crucial for effective programming in JavaScript. This section explores the fundamental data structures in JavaScript, including **Arrays**, **Objects**, **Sets**, and **Maps**, providing explanations and code block examples for each.

### Arrays

Arrays in JavaScript are used to store multiple values in a single variable. They are list-like objects that come with a variety of methods to perform traversal and mutation operations.

```javascript
// Creating an array
const fruits = ["Apple", "Banana", "Cherry"];

// Accessing an element
console.log(fruits[1]); // Output: Banana

// Adding an element to the end
fruits.push("Durian");
console.log(fruits); // Output: ["Apple", "Banana", "Cherry", "Durian"]

// Removing the last element
fruits.pop();
console.log(fruits); // Output: ["Apple", "Banana", "Cherry"]
```

### Objects

Objects in JavaScript are collections of properties, where each property is a key-value pair. Objects are used to store structured data and complex entities.

```javascript
// Creating an object
const person = {
    firstName: "John",
    lastName: "Doe",
    age: 30,
};

// Accessing properties
console.log(person.firstName); // Output: John

// Adding a new property
person.job = "Developer";
console.log(person.job); // Output: Developer

// Deleting a property
delete person.age;
console.log(person); // Output: { firstName: "John", lastName: "Doe", job: "Developer" }
```

### Sets

A Set is a collection of unique values. Unlike arrays, sets do not allow duplicate values, making them ideal for storing unique items.

```javascript
// Creating a Set
const mySet = new Set([1, 2, 3, 4, 4, 5]);

// Adding an element
mySet.add(6);
console.log(mySet); // Output: Set(6) {1, 2, 3, 4, 5, 6}

// Checking for existence
console.log(mySet.has(3)); // true

// Removing an element
mySet.delete(2);
console.log(mySet); // Output: Set(5) {1, 3, 4, 5, 6}

// Clearing the Set
mySet.clear();
console.log(mySet); // Output: Set(0) {}
```

### Maps

Maps are key-value pairs where keys can be of any type. Unlike objects, which have strings or symbols as keys, Maps allow for keys of any data type.

```javascript
// Creating a Map
const myMap = new Map([
    ['key1', 'value1'],
    ['key2', 'value2'],
]);

// Setting a key-value pair
myMap.set('key3', 'value3');
console.log(myMap.get('key3')); // Output: value3

// Checking for a key
console.log(myMap.has('key2')); // true

// Deleting a key-value pair
myMap.delete('key1');
console.log(myMap); // Output: Map(2) {"key2" => "value2", "key3" => "value3"}

// Getting the size
console.log(myMap.size); // 2

// Clearing the Map
myMap.clear();
console.log(myMap); // Output: Map(0) {}
```

In JavaScript, to check for the presence of an item in a `Set` or a key in a `Map`, you use the `.has()` method. This method returns `true` if the element (for a Set) or key (for a Map) exists in the collection, and `false` otherwise. This is somewhat analogous to the `in` keyword in Python for checking membership in sets and dictionaries.

To add a key-value pair to a `Map` in JavaScript, you use the `.set(key, value)` method. Here is a simple example demonstrating how to add elements to a `Map`:

```javascript
// Create a new Map
const myMap = new Map();

// Add key-value pairs to the map
myMap.set('key1', 'value1');
myMap.set('key2', 'value2');
myMap.set('key3', 'value3');

// Log the entire Map
console.log(myMap);
// Output: Map { 'key1' => 'value1', 'key2' => 'value2', 'key3' => 'value3' }

// You can also chain .set() methods
myMap.set('key4', 'value4').set('key5', 'value5');

// Check the updated Map
console.log(myMap);
// Output: Map { 'key1' => 'value1', 'key2' => 'value2', 'key3' => 'value3', 'key4' => 'value4', 'key5' => 'value5' }
```

The `.set(key, value)` method allows you to add new key-value pairs to the map or update the value of an existing key. It returns the `Map` object itself, hence allowing for method chaining as shown in the example.

#### Checking Membership in a Set

For a `Set`, you use `.has(value)` to check if a value is in the set.

```javascript
const numSet = new Set([1, 2, 3, 4, 5]);

if (numSet.has(3)) {
    console.log("3 is in the set");
} else {
    console.log("3 is not in the set");
}
```

#### Checking Membership in a Map

For a `Map`, you also use `.has(key)` to check if a key is in the map.

```javascript
const numMap = new Map([
    [1, "one"],
    [2, "two"],
    [3, "three"],
]);

if (numMap.has(3)) {
    console.log("3 is a key in the map");
} else {
    console.log("3 is not a key in the map");
}
```

These methods provide a straightforward way to check for the presence of keys or values in JavaScript's `Set` and `Map` objects, respectively, allowing for efficient conditional checks similar to Python's `in` syntax but with the appropriate method call for the specific collection type.

---
## More Advanced Data Structures
JavaScript, unlike Python, does not come with built-in implementations of data structures such as heaps, stacks, or queues as part of its standard library. In Python, the `collections` module provides a `deque` class for queue and stack implementations, and the `heapq` module provides functions that implement a heap queue on a list. JavaScript's standard library is more minimalistic in this regard, focusing on a set of built-in objects like `Array`, `Set`, `Map`, etc., without direct analogs for these more specialized structures.

### Workarounds and Alternatives

- **Stacks and Queues:** While JavaScript does not have built-in classes for stacks and queues, their functionality is easily replicated with arrays. The array methods `push()`, `pop()`, and `shift()` allow arrays to be used as stacks or queues directly.
  
- **Heaps:** For heap implementations, JavaScript developers typically need to implement their own heap logic or use a third-party library. The logic involves creating a class or function that manages the heap property on an array. The lack of a built-in heap structure means that for specific applications like priority queues, developers must either implement their own or rely on external libraries.

### External Libraries

For those looking for out-of-the-box heap implementations in JavaScript, there are several high-quality libraries and npm packages available, such as:
- **`heap`**: A popular npm package that provides a simple binary heap implementation, allowing for both min-heap and max-heap configurations.
- **`collections`**: This library offers a variety of data structures, including heaps, and might be used to fill in some of the gaps in the native JavaScript environment.

### Custom Implementation Example

Here’s a brief example of how you might implement a simple min-heap in JavaScript, demonstrating that while direct support isn't provided by JavaScript, manual implementation is straightforward:

```javascript
class MinHeap {
  constructor() {
    this.heap = [];
  }

  getParentIndex(i) { return Math.floor((i - 1) / 2); }
  getLeftChildIndex(i) { return 2 * i + 1; }
  getRightChildIndex(i) { return 2 * i + 2; }

  insert(key) {
    this.heap.push(key);
    let index = this.heap.length - 1;
    let parent = this.getParentIndex(index);

    while (index > 0 && this.heap[parent] > this.heap[index]) {
      [this.heap[parent], this.heap[index]] = [this.heap[index], this.heap[parent]];
      index = parent;
      parent = this.getParentIndex(index);
    }
  }

  // Methods for extractMin, heapify, etc., would also be necessary
}
```


---

## Control Flow and Error Handling

Control flow in JavaScript dictates the order in which the computer executes statements in a script. Error handling ensures that your program can gracefully respond to unexpected situations. This section explores key concepts including conditional statements, loops, and try-catch error handling.

### Conditional Statements

Conditional statements allow you to run different blocks of code based on different conditions.

#### if-else

```javascript
const age = 18;

if (age >= 18) {
    console.log("You are an adult.");
} else {
    console.log("You are a minor.");
}
```

#### switch

```javascript
const day = new Date().getDay();

switch (day) {
    case 0:
        console.log("It's Sunday");
        break;
    case 1:
        console.log("It's Monday");
        break;
    // Add cases for other days
    default:
        console.log("Looking forward to the Weekend");
}
```

### Loops

Loops are used to execute a block of code a number of times, as long as a specified condition is true.

#### for Loop

```javascript
for (let i = 0; i < 5; i++) {
    console.log(`This is iteration number ${i}`);
}
```

*for of loop*
```javascript 
const nums = [1, 2, 3, 4, 5];

for (const num of nums) {
    console.log(num);
}
```
#### while Loop

```javascript
let i = 0;
while (i < 5) {
    console.log(`This is iteration number ${i}`);
    i++;
}
```

### Error Handling

Error handling in JavaScript is achieved using the try-catch statement. It allows you to test a block of code for errors while it is being executed.

```javascript
try {
    // Code to try
    nonExistentFunction();
} catch (error) {
    // Code to run if an error occurs
    console.log(error.message);
} finally {
    // Code that is always executed regardless of an exception occurs
    console.log("This always runs");
}
```


---

Following the "Control Flow and Error Handling" section, a crucial next part of your JavaScript overview could be **"Functions and Scope"**. This section is essential because functions are the building blocks of JavaScript code, enabling modular, reusable code writing. Scope, on the other hand, dictates the accessibility of variables and functions in different parts of your code. Understanding both concepts is key to mastering JavaScript.

---

## Functions and Scope

In JavaScript, functions are used to define reusable blocks of code. Scope determines the accessibility of these functions and variables. This section covers the creation and use of functions, different types of functions, and the concept of scope.

### Functions

Functions allow you to encapsulate a task. They can take parameters and can return a value.

#### Function Declaration

```javascript
function greet(name) {
    return `Hello, ${name}!`;
}

console.log(greet("Alice")); // Output: Hello, Alice!
```

#### Function Expression

```javascript
const square = function(number) {
    return number * number;
};

console.log(square(4)); // Output: 16
```

#### Arrow Functions

Introduced in ES6, arrow functions offer a concise syntax for writing function expressions.

```javascript
const add = (a, b) => a + b;

console.log(add(2, 3)); // Output: 5
```

### Scope

Scope in JavaScript refers to the current context of code, which determines the visibility or accessibility of variables and functions.

#### Global Scope

Variables defined outside any function have global scope and are accessible from anywhere in the code.

```javascript
let color = "blue";

function getColor() {
    return color; // Accessible because color is in the global scope
}

console.log(getColor()); // Output: blue
```

#### Local (Function) Scope

Variables declared within a function are local to that function and cannot be accessed from outside.

```javascript
function greet() {
    let greeting = "Hello, World!";
    return greeting;
}

console.log(greet()); // Output: Hello, World!
// console.log(greeting); // Uncaught ReferenceError: greeting is not defined
```

#### Block Scope

Introduced with ES6, `let` and `const` provide block-level scope, restricting the visibility of a variable to the block in which it is declared.

```javascript
if (true) {
    let blockScopedVariable = "I'm only accessible in this block";
    console.log(blockScopedVariable); // Output: I'm only accessible in this block
}
// console.log(blockScopedVariable); // Uncaught ReferenceError: blockScopedVariable is not defined
```

### Closure

A closure is a function that remembers the variables from the place where it is defined, regardless of where it is executed.

```javascript
function makeAdder(x) {
    return function(y) {
        return x + y;
    };
}

const addFive = makeAdder(5);
console.log(addFive(2)); // Output: 7
```

---

This block of code encompasses various JavaScript functionalities, primarily focusing on **DOM Manipulation**, **Event Handling**, and **Web Storage**. Here's how you might categorize and describe these snippets in a section:

---

## DOM Manipulation, Event Handling, and Web Storage

This section highlights practical JavaScript applications for dynamic webpage interaction and data management. It covers manipulating the Document Object Model (DOM), handling user events, and utilizing web storage for persistent data.

### DOM Manipulation

Manipulating the DOM allows you to change webpage content and styles dynamically. This includes changing element styles, updating text content, and modifying attributes.

```javascript
// Change header color on click
document.querySelector("header").addEventListener("click", function () {
    this.style.backgroundColor = "#4CAF50";
});

// Dynamically update the About Section
window.addEventListener("load", function () {
    document.getElementById("about").querySelector("p").innerText =
        "Updated information about us and our projects.";
});

// Add alt text to images dynamically
document.querySelectorAll("img").forEach(function (img) {
    img.setAttribute("alt", "Descriptive text");
});
```

### Event Handling

Event handling enables interactive web pages by responding to user inputs or actions, such as clicks, mouseovers, and form submissions.

```javascript
// Toggle Navigation Menu Visibility on Mobile
document.querySelector("nav").addEventListener("click", function () {
    const ul = this.querySelector("ul");
    ul.style.display = ul.style.display === "none" ? "block" : "none";
});

// Validate Email Form Submission
document.querySelector("form").addEventListener("submit", function (event) {
    const email = document.getElementById("email").value;
    if (!/\S+@\S+\.\S+/.test(email)) {
        alert("Please enter a valid email address.");
        event.preventDefault();
    }
});

// Change style of header on hover
document.querySelector("header").addEventListener("mouseover", function () {
    this.style.color = "#FFFFFF";
    this.style.backgroundColor = "#333333";
});
document.querySelector("header").addEventListener("mouseout", function () {
    this.style.color = "#000000";
    this.style.backgroundColor = "#f0f0f0";
});
```

### Web Storage

Web Storage API allows storing data locally in the user's browser, enabling persistence of data across sessions.

```javascript
// Save message to local storage
document.querySelector("form").addEventListener("submit", function (event) {
    const message = document.getElementById("message").value;
    localStorage.setItem("message", message);
    event.preventDefault(); // For demonstration
    alert("Message saved locally.");
});

// Retrieve and display message from local storage on load
window.addEventListener("load", function () {
    const message = localStorage.getItem("message");
    if (message) {
        document.getElementById("message").value = message;
    }
});
```

