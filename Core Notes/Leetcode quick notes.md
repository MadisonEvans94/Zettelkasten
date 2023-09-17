
### 1sum 
- use enumerate 
- create an empty dictionary
- iterate through each item of array
- if the difference between that value and the target **is not** in the dictionary, then add it with
```python
nums_dict[n] = i
```
- if the difference between that value and the target **is not** in the dictionary, then return answer with 
```python 
return [nums_dict[target-n],i]
```