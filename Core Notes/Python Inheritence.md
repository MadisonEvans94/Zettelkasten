#seed 
upstream:

---

**video links**: 

---

# Brain Dump: 


--- 


Inheritance is a core concept in object-oriented programming (OOP) that allows you to define a new class based on an existing class. The new class inherits attributes and behaviors (methods) from the existing class.

## Basic Inheritance

Let's start with a basic example:

```python
class Animal:
    def make_sound(self):
        print("Some generic sound")

class Dog(Animal):
    pass

d = Dog()
d.make_sound()  # Output will be "Some generic sound"
```

Here, `Dog` is a subclass that inherits from the `Animal` superclass. Since `Dog` doesn't override the `make_sound` method, it inherits the one from `Animal`.

## Overriding Methods

Subclasses can provide specific implementations of methods that are already defined in their superclasses. This is known as method overriding.

```python
class Animal:
    def make_sound(self):
        print("Some generic sound")

class Dog(Animal):
    def make_sound(self):
        print("Woof")

d = Dog()
d.make_sound()  # Output will be "Woof"
```

## The `super()` Function

The `super()` function in Python is used to call a method in a superclass from the subclass that inherits from it. This is especially useful when you override a method but still want to call the original method.

```python
class Animal:
    def make_sound(self):
        print("Some generic sound")

class Dog(Animal):
    def make_sound(self):
        super().make_sound()
        print("Woof")

d = Dog()
d.make_sound()
# Output will be:
# Some generic sound
# Woof
```

## `super()` in PyTorch Models

When defining neural network models in [[Pytorch]], you'll often subclass `nn.Module`. This superclass provides a lot of built-in functionalities (like `.forward()` and `.backward()`) that your neural network model will need.

```python
import torch.nn as nn

class SimpleNNWithBatchNorm(nn.Module):
    def __init__(self):
        super(SimpleNNWithBatchNorm, self).__init__()
        # Your code for layers and operations
```

### Why `super(SimpleNNWithBatchNorm, self).__init__()`?

Here, `super(SimpleNNWithBatchNorm, self).__init__()` calls the `__init__` method of the `nn.Module` superclass. This is crucial because the superclass's constructor does a lot of behind-the-scenes work that makes the neural network functional. If you don't call it, you'll miss out on those initializations and your class won't work as expected.

The `super(SimpleNNWithBatchNorm, self).__init__()` syntax is especially important in the context of multiple inheritance, where a class can inherit from multiple superclasses. PyTorch relies on this to correctly initialize the underlying C++ classes.





