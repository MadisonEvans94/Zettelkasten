#seed 
upstream: [[Javascript]]

---

**links**: 

---

The `??` operator in JavaScript and TypeScript is known as the Nullish Coalescing Operator. It is used to provide a default value for a variable that is either `null` or `undefined`. It's a logical operator that returns the right-hand operand when the left-hand operand is `null` or `undefined`, and otherwise returns the left-hand operand.

In the context of your example:

```typescript
const data: number[] = labels.map(
    (label) => aggregatedData[label].categories[category] ?? 0
);
```

Here, `aggregatedData[label].categories[category] ?? 0` checks if `aggregatedData[label].categories[category]` is `null` or `undefined`. If it is, the expression evaluates to `0`, providing a default value. If `aggregatedData[label].categories[category]` contains a value that is not `null` or `undefined`, it returns that value instead.

This operator is particularly useful for setting default values in a concise manner, avoiding more verbose conditional statements like `if-else` or ternary operators for these specific cases.



