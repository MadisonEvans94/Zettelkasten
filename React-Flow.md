#seed 
upstream:

---

**links**: 
- [code sandbox](https://codesandbox.io/s/zen-grass-zn3f2r?file=/src/Flow.tsx)
---

Brain Dump: 

--- 
# Anatomy of a Flow 


![[Screenshot 2024-01-09 at 8.36.07 PM.png]]

---
## BasicFlow Component 
```tsx
import { useCallback } from "react";
import ReactFlow, {
Node,
addEdge,
Background,
Edge,
Connection,
useNodesState,
useEdgesState
} from "reactflow";
import CustomNode from "./CustomNode";
import "reactflow/dist/style.css";

const initialNodes: Node[] = [
	{
		id: "1",
		type: "input",
		data: { label: "Node 1" },
		position: { x: 250, y: 5 }
	},
	{ id: "2", data: { label: "Node 2" }, position: { x: 100, y: 100 } },
	{ id: "3", data: { label: "Node 3" }, position: { x: 400, y: 100 } },
	{ 
		id: "4", 
		type: "custom", 
		data: { label: "Custom Node" }, 
		position: { x: 400, y: 200 }
	}
];

const initialEdges: Edge[] = [
	{ id: "e1-2", source: "1", target: "2" },
	{ id: "e1-3", source: "1", target: "3" }
];

const nodeTypes = {
	custom: CustomNode
};

const BasicFlow = () => {

	const [nodes, onNodesChange] = useNodesState(initialNodes);
	const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
	const onConnect = useCallback(
		(params: Edge | Connection) => setEdges((els) => addEdge(params, els)),
		[setEdges]
	);

	return (

		<ReactFlow
			nodes={nodes}
			edges={edges}
			onNodesChange={onNodesChange}
			onEdgesChange={onEdgesChange}
			onConnect={onConnect}
			nodeTypes={nodeTypes}
			fitView
		>			
			<Background />
		</ReactFlow>
	);
};

export default BasicFlow;
```
### `useNodesState` and `onNodesChange`
The `useNodesState` is a custom hook provided by React Flow that manages the state of nodes in your flow diagram. When you call this hook, it returns an array containing the current nodes, a setter function (which is not used in your current code), and a change handler function `onNodesChange`.

The `onNodesChange` is a function that gets called whenever a node is changed, for example, when you drag a node to a new position. This function takes care of updating the nodes' state internally to reflect these changes.

### `useEdgesState` and `onEdgesChange`
Similarly, `useEdgesState` is another custom hook for managing the state of edges. It also returns the current edges, a setter function `setEdges`, and a change handler `onEdgesChange`.

The `onEdgesChange` method is the equivalent for edges of what `onNodesChange` is for nodes. It's called whenever an edge is added, removed, or updated.

### `onConnect` Callback
The `onConnect` function is a callback that you define to handle what happens when a new connection is made between nodes. When you drag a link from one node and drop it to another, `onConnect` is invoked.

Here's a step-by-step breakdown of the `onConnect` callback:

1. `onConnect` is called with either an `Edge` or `Connection` object as its parameter. This object contains information about the source and target of the new connection.
   
2. Inside `onConnect`, the `addEdge` function from React Flow is called. This function takes the `params` (the connection object) and the current edges (`els`) as arguments.

3. `addEdge` adds the new edge to the current list of edges and returns a new array that includes this new edge.

4. `setEdges` is then called with the new array of edges, which updates the state of the edges in the component.

> To extend the `onConnect` logic to, for instance, make an HTTP request or update the state, you would modify the `onConnect` function. Here's an example of how you might do that:

```jsx
const onConnect = useCallback(
	(params: Edge | Connection) => {
		// Add the new edge
		setEdges((els) => addEdge(params, els));

		// Here you can add logic to perform an HTTP request
		// Or update some other state in response to the new connection
		const newEdge = { ...params, yourCustomData: 'example' };
		fetch('your-api-endpoint', {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
			},
			body: JSON.stringify(newEdge),
		}).then(response => {
			// Handle response
		}).catch(error => {
			// Handle error
		});
	},
	[setEdges]
);
```

---
## CustomNode 

```tsx
import { memo } from "react";
import { Handle, NodeProps, Position } from "reactflow";
const CustomNode = ({
	data,
	isConnectable,
	targetPosition = Position.Top,
	sourcePosition = Position.Bottom
}: NodeProps) => {
	return (
		<>
			<Handle
				type="target"
				position={targetPosition}				
				isConnectable={isConnectable}
			/>
			{data?.label}
			<Handle
				type="source"
				position={sourcePosition}
				isConnectable={isConnectable}
			/>
		</>
	);
};

CustomNode.displayName = "CustomNode";
export default memo(CustomNode);
```

### `memo`
In the provided code, `memo` is a higher-order component from React that memoizes your component. This means that React will render the component and memoize the result. Before the next render, if the new props are the same as the previous props, React will reuse the memoized result, skipping the rendering phase for the component. This can lead to performance improvements, especially for components that render often with the same props.

### `CustomNode` Component Explanation
The `CustomNode` component is a functional component that is designed to be used as a custom node type in React Flow. It receives `NodeProps` which are the props passed down from React Flow to control the node's behavior and appearance.

Here's a breakdown of its parts:

- **Handles**: In React Flow, `Handle` components are used to create connection points on the nodes where edges can be dragged from or dropped to. The `CustomNode` component includes two handles:
  - A `target` handle, which means it can accept incoming connections. Its position is determined by `targetPosition`.
  - A `source` handle, which allows outgoing connections. Its position is determined by `sourcePosition`.
  
- **isConnectable**: This is a prop that determines if the node's handles are currently connectable. If `false`, you won't be able to drag a new edge from or drop a new edge to this node.

- **data**: The `data` prop is an object that holds custom data for the node. In this case, it's being used to display a label (`data.label`).

- **targetPosition** and **sourcePosition**: These are props with default values set to `Position.Top` and `Position.Bottom`, respectively. They are imported from `reactflow` and used to position the handles on the node. If the parent component that uses `CustomNode` doesn't pass these props, the handles default to these positions.

- **`<Handle>` Components**: Each `Handle` component is rendered with a specific `type` ("target" for incoming edges and "source" for outgoing edges) and a `position`. The `isConnectable` prop is also passed to determine if you can create connections with this handle.

- **`displayName`**: This is a property set on the component to give it a name in the React DevTools, which can help with debugging.

Here's an example of how the `memo` function is wrapping the `CustomNode` component:

```jsx
export default memo(CustomNode);
```

By exporting the `CustomNode` with `memo`, you ensure that this component only re-renders when its props change, potentially reducing the number of renders and improving the performance of your application, especially if the `CustomNode` is used frequently within your flow.


---

Certainly! In React Flow, the `Node` and `Edge` objects are fundamental elements that define the structure and connections within the flow diagram. Here's an overview of each:

### Node Object Anatomy
A Node object represents a single element within the flow diagram. It typically has the following properties:

- **id**: A unique string that identifies the node. It's used to reference the node within the React Flow instance.
- **type**: Specifies the type of the node. By default, it can be `"input"`, `"default"`, `"output"`, or any custom type you define. Custom types are used to render different components for different nodes. 
> see [[The Difference Between 'Input', 'Default', and 'Output' Types]] for more
- **data**: An object that holds the custom data for the node. This could be anything relevant to the node, such as labels or form inputs.
- **position**: An object with `x` and `y` properties that determine the node's position on the canvas.
- **style**: (Optional) A CSS properties object that defines the inline style for the node's wrapper element.
- **className**: (Optional) A string that allows you to apply custom CSS classes to the node's wrapper element.
- **sourcePosition**: (Optional) Defines the position of the outgoing handle. It can be `'top'`, `'right'`, `'bottom'`, or `'left'`.
- **targetPosition**: (Optional) Defines the position of the incoming handle, with the same possible values as `sourcePosition`.

Here's an example of a Node object:

```javascript
{
  id: '1',
  type: 'input',
  data: { label: 'Node 1' },
  position: { x: 250, y: 5 },
  style: { border: '1px solid #777', padding: '10px' },
  className: 'my-custom-node'
}
```

### Edge Object Anatomy
An Edge object connects two nodes and represents a dependency or flow between them. An Edge has these properties:

- **id**: A unique identifier for the edge.
- **source**: The `id` of the source node where the edge starts.
- **target**: The `id` of the target node where the edge ends.
- **type**: (Optional) The type of the edge. You can define custom edge types to render different kinds of edges.
- **label**: (Optional) A label for the edge, which can be displayed on the diagram.
- **animated**: (Optional) A boolean that, when `true`, shows an animation along the edge to indicate flow direction.
- **style**: (Optional) An object that defines the CSS styles for the edge line.
- **arrowHeadType**: (Optional) The type of arrowhead to use at the end of the edge.

An example Edge object might look like this:

```javascript
{
  id: 'e1-2',
  source: '1',
  target: '2',
  type: 'smoothstep',
  label: 'Edge from Node 1 to Node 2',
  animated: true,
  style: { stroke: '#f6ab6c' },
  arrowHeadType: 'arrowclosed'
}
```

These objects are quite flexible and can be extended with additional properties if needed to suit the specific needs of your flow diagram. In your React Flow application, you'll manipulate these Node and Edge objects to define and update the diagram's structure and appearance.