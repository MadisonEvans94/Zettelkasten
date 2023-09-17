
###### Upstream: 
###### Siblings: [[Priority Queue]], [[heap]]
#incubator 

2023-05-25
18:22


### Main Takeaways

This Python function pushes a new item onto the heap, preserving the heap invariant by swapping the new item with its parent until it reaches a place where it is less than or equal to its parent.

Note that this is a simplified illustration and doesn't include all optimizations present in the actual C implementation of `heapq.heappush()`. It's also designed for a min-heap, where parent elements are less than or equal to their children. To adapt it for a max-heap, you'd need to adjust the heap property comparison.

### Example Code: 

```python
def heappush(heap, item):
    # Append the item to the end of the heap
    heap.append(item)

    # Get the index of the last element (the one we just appended)
    idx = len(heap) - 1

    # Bubble-up until heap property is restored
    while idx > 0:
        # Compute the parent's index
        parent_idx = (idx - 1) // 2

        # If the heap property is not violated, break out of the loop
        if heap[parent_idx] <= heap[idx]:
            break

        # Swap the current element with its parent
        heap[idx], heap[parent_idx] = heap[parent_idx], heap[idx]

        # Update the current index to be the parent's index
        idx = parent_idx


```