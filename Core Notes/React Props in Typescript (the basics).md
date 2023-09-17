#incubator 
###### upstream: [[Typescript]], [[React]]

"use types when building applications and interfaces when building libraries" [video](https://www.youtube.com/watch?v=KpA6oEaCHtk&list=PLC3y8-rFHvwi1AXijGTKM0BKtHzVC-LSK&index=3&ab_channel=Codevolution)

### Syntax: 

*Consider the example below of a Component that is using props*

```tsx
type LayoutProps = {
	sidebar: string;
	dash: string;
};

const DashboardLayout = () => {
	return <Layout sidebar="sidebar" dash="dash" />;
};

const Layout = (props: LayoutProps) => {
	const [sidebar, dash] = [props.sidebar, props.dash];
	return (
		<div className="w-full h-full fixed bg-pink-500 flex flex-row">
			<div className="bg-blue-400 min-w-[200px] w-[20%]">{sidebar}</div>
			<div className="bg-orange-300 h-full w-full p-10">
				<div className="h-full w-full bg-yellow-200">{dash}</div>
			</div>
		</div>
	);
};

export default DashboardLayout;
```

The process flow in typescript for creating components that use props should be as follows: 
1. **Make a `type` object** that defines the structure of the `props` object of the component in question 
2. **Define the property types** of the properties within the `type` object 
3. **Declare the type of the `props` object** when defining the component, i/e:
```ts
const Layout = (props: LayoutProps) => {...}
```

### Destructuring Props: 

### Working With An Array: 

Array of objects: 

```tsx
type PersonListProps = {
	names: {
		first: string
		last: string
	}[]
}

export const PersonList = (props: PersonListProps) => {
	return (
		<div>
			{props.names.map((name) => {
				return(
					<h2 key={name.first}> 
						{name.first} {name.last}
					</h2>
				)
			})}
		</div>
	)
}
```

### More Advanced Examples: 

**union of string literals**
```tsx
type StatusProps = {
	status = "loading" | "paused" | "active"
}
```

the status attribute of the `StatusProps` type is a string, but can only be the value `"loading"`, `"paused"`, or `"active"`

**passing children**

```tsx
function App() {
	return (
		<div>
			<Heading>Placeholder text</Heading>
		</div>
	)
}
```

*^This will error unless we do the following to the `Heading.tsx` component ...* 
```tsx 
type HeadingProps = {
	children: string
}

const Heading = (props:HeadingProps) => {
	return <h1>{props.children}</h1>
}
```

**Passing a Child that is another Component**

```tsx
function App() {
	return (
		<div>
			<Oscar>
				<Heading>Placeholder text</Heading>
			</Oscar>
		</div>
	)
}
```

*^The above will error because we haven't defined the structure for the children of the `Oscar` component. To fix this, let's do the following to the `Oscar.tsx `component...*

```tsx
type OscarProps = {
	children: React.ReactNode
}

export const Oscar = (props: OscarProps) => {
	return <div>{props.children}</div>
}
```

Notice that in order to pass other react components as children, we need to define the children type as `React.ReactNode`. For more info, see [[React Node]]

**Optional Inclusion**

*In order to let React know that a particular prop is optional, we should add a `?` at the end of the property name that we want to be optional. See the following example...*

```tsx
type GreetProps = {
	name: string
	messageCount?: number
	isLoggedIn: boolean
}
```
