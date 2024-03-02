#seed 
upstream:

---

**links**: 

---

Brain Dump: 

--- 

In React Flow, the `input`, `default`, and `output` types refer to predefined node behaviors and styles that are commonly used in flow-based diagrams or node-based editors.

### Input Node Type
- **Purpose**: Input nodes are typically used as starting points in a flow. They are sources of data or triggers that initiate the process modeled by the flow diagram.
- **Behavior**: They usually don't have incoming edges because they are at the beginning of a workflow. They can only have outgoing edges.
- **Style**: Input nodes might be styled differently to stand out as the starting points in the diagram.

### Default Node Type
- **Purpose**: Default nodes are the standard nodes used to represent most of the steps or operations within the flow. They can represent functions, actions, or decisions.
- **Behavior**: They can have both incoming and outgoing edges, representing the flow from one operation to the next.
- **Style**: They typically have a standard look that is consistent throughout the diagram, differentiating them from the specialized input and output nodes.

### Output Node Type
- **Purpose**: Output nodes are typically used as endpoints in a flow. They represent the result or the final output of the process.
- **Behavior**: They usually don't have outgoing edges as they signify the end of a process flow. They can have multiple incoming edges.
- **Style**: Output nodes might be styled to indicate that they are the conclusion of the flow, with different colors or shapes to signify the end of the process.

In React Flow, you can also define custom node types, which allow you to create your own components with unique behaviors and styles. Custom nodes can be designed to fit any specific needs that go beyond the provided `input`, `default`, and `output` types. For instance, in your code, there's a `custom` node type associated with the `CustomNode` component that you can design as you see fit for your flow diagram.





