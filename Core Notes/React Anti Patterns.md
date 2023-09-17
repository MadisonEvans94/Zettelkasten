#incubator 
###### upstream: [[React]]

### Origin of Thought:
- improve software development by being aware of antipatterns

### Underlying Question: 
- What are the top most common anti patterns to avoid in React? 

### Solution/Reasoning: 
1.  **Calling hooks conditionally:** The order in which hooks are called should be the same on every render. This means hooks should not be called inside loops, conditions, or nested functions.

Anti-pattern:

```jsx
if (condition) {   
	const [state, setState] = useState(initialState); // wrong 
}
```


Correct way:


```jsx
const [state, setState] = useState(condition ? initialState : otherState); // right
```


2.  **Infinite loops with `useEffect`:** You need to be careful with the dependency array in `useEffect`. If you use a value inside `useEffect` that changes too often, it could create an infinite loop.

Anti-pattern:

```jsx
const [value, setValue] = useState(0);  
useEffect(() => {   
	setValue(value + 1);  // wrong - causes infinite loop 
}, [value]);
```


Correct way:

```jsx
const [value, setValue] = useState(0);  
useEffect(() => {   
	const timer = setTimeout(() => {     
		setValue(value + 1);  
	}, 1000);      
	return () => clearTimeout(timer);  // cleanup 
}, []);
```


3.  **Large functional components:** Just like with class components, having large functional components can make them hard to understand and maintain. Try to split them into smaller, reusable functional components.

4.  **Relying only on `useEffect` for all side-effects:** Not all side effects belong in `useEffect`. Sometimes, it makes more sense to perform a side effect during an event handler.

Anti-pattern:

```jsx
function MyComponent() {   
	const [count, setCount] = useState(0);    
	useEffect(() => {     
		document.title = `Count is ${count}`;  // wrong - updates on every render   
	});    
	return <button onClick={() => setCount(count + 1)}>Increase Count</button>; 
}
```


Correct way:


```jsx
function MyComponent() {   
	const [count, setCount] = useState(0);    
	const handleClick = () => {     
		const newCount = count + 1;     
		document.title = `Count is ${newCount}`;  // right - only updates on click     
		setCount(newCount);   
	};    
	return <button onClick={handleClick}>Increase Count</button>; 
}
```

5.  **Using too many states:** If several states are always updated together, consider using one state object instead.

Anti-pattern:


```jsx
const [name, setName] = useState(''); 
const [age, setAge] = useState(0); 
const [city, setCity] = useState('');
```

Correct way:


```jsx
const [form, setForm] = useState({ name: '', age: 0, city: '' });
```


6.  **Ignoring the rule of hooks:** The rule of hooks should be followed to ensure hooks work as expected. Rules include only calling hooks at the top level of your React function and only calling hooks from React functions or custom hooks.


Remember that every situation is unique, and sometimes, it might make sense to deviate from these guidelines. They are not strict rules, but rather best practices that can help improve your React code in most scenarios.

### Examples (if any): 


### Additional Thoughts: 
- [[What are side effects and why are they bad?]]
