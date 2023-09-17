#seed 
###### upstream: [[Typescript]]


### Click Event 

Let's say we have a `Button` component that we want to accept a click event as prop and send to html

```tsx
export const Button = () => {
	return <button>Click</button>
}
```

Let's begin by adding the type at the top as follows... 

```tsx
type ButtonProps = {
	handleClick: () => void
}

export const Button = (props: ButtonProps) => {
	return <button onClick={props.handleClick}>Click</button>
}
```

Now back in the App component, we can write as the following: 

```tsx 
import { Button } from './components/Button'

function App() {
	return (
		<div>
			<Button handleClick={() => {
				console.log('Button clicked')
				}}
			/>
		</div>
	)
}
```

As you can see, the `handleClick: () => void` line simply describes the structure of the funciton (what type of return value is expected). But the actual implementation within the App.js component has a more concrete implementation 

Another variation of this mouse event approach uses the following syntax: 

```tsx
type ButtonProps = {
	handleClick: (event: React.MouseEvent<HTMLButtonElement>) => void
}
```


### Inputs: 

In React, when a user has input component, the component needs 2 props: input value and onChange handler

```tsx
type InputProps = {
	value: string 
	handleChange: (event: React.ChangeEvent<HTMLInputElement>) => void
}

export const Input = (props: InputProps) => {
	return <input type='text' value={props.value} onChange={props.handleChange}/>
}
```

back in `App.tsx`, we would have the following: 


```tsx 
import { Button } from './components/Button'
import { Input } from './components/Input'

function App() {
	return (
		<div>
			<Button handleClick={() => {
				console.log('Button clicked')
				}}
			/>
			<Input value='' handleChange={event => console.log(event)}
		</div>
	)
}
```