#seed 
upstream: [[Design Patterns]]

---

**video links**: 

---

# Brain Dump: 

## What is the Template Method Pattern?

The pattern consists of two essential parts:

1. **Abstract Class**: This class contains a template method that defines the skeleton of an algorithm, calling both concrete methods and abstract methods that must be implemented by subclasses.

2. **Concrete Subclasses**: These classes provide the specific implementation for the abstract methods defined in the abstract class, without altering the overall structure of the algorithm.

### Example

```dart
abstract class AbstractClass {
  // Template method
  void templateMethod() {
    operation1();
    operation2();
  }

  void operation1(); // Abstract method
  void operation2(); // Abstract method
}

class ConcreteClass extends AbstractClass {
  @override
  void operation1() {
    // Implementation
  }

  @override
  void operation2() {
    // Implementation
  }
}
```

---

## When and Why is it Used?

### When

- When you want to let clients extend only particular steps of an algorithm.
- When you have several classes that contain almost identical code for certain operations, and you want to avoid duplicating that code.

### Why

- **Flexibility**: By defining the structure in the base class and allowing subclasses to provide specific implementations, the pattern offers great flexibility without altering the overall structure.
- **Code Reusability**: Encourages code reusability by extracting common code into a base class, allowing variations in subclasses.
- **Maintainability**: Changes to the overall algorithm can be made in the base class without affecting subclass implementations.
---

## Where is it Commonly Seen in Front-End Applications?

1. **UI Frameworks**: Many UI frameworks, including Flutter, use the Template Method Pattern to provide a consistent structure for components or widgets while allowing customization through subclassing.

2. **Form Handling**: Some libraries use this pattern to provide a common structure for form validation or submission, allowing developers to customize specific steps like data validation or transformation.

3. **Lifecycle Methods**: Many front-end frameworks provide lifecycle methods that can be seen as a form of the Template Method Pattern. You can define certain behaviors at specific stages of a component's lifecycle without changing the overall lifecycle flow.

---

## Dart Example: Template Method Pattern in Flutter

In the context of Flutter, the Template Method Pattern is used to define a consistent structure for widgets, particularly in the creation of `StatefulWidget` classes. Let's examine the following example:

```dart
class Counter extends StatefulWidget {
  @override
  _CounterState createState() => _CounterState(); // Template Method
}

class _CounterState extends State<Counter> {
  int count = 0;

  void increment() {
    setState(() {
      count++;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Text('Count: $count'),
        ElevatedButton(onPressed: increment, child: Text('Increment')),
      ],
    );
  }
}
```

### How This Code Illustrates the Template Method Pattern

1. **Abstract Class (`StatefulWidget`)**: The `StatefulWidget` class defines a method `createState` that acts as the template method. This method returns a `State` object and sets the expectation that subclasses must implement this method to create the associated `State`.

2. **Concrete Subclass (`_CounterState`)**: The `_CounterState` class provides the specific implementation for the `State` associated with the `Counter` widget. It includes the concrete implementation of the `build` method, defining how the widget should be rendered.

3. **Template Method (`createState`)**: The `createState` method in the `Counter` class acts as the template method, providing the structure of how the state is created. Subclasses (like `_CounterState`) provide the specific implementation of the `State`.

This example demonstrates how the Template Method Pattern is applied in Flutter to create reusable and customizable widgets. By defining a template method (`createState`) in the abstract `StatefulWidget` class and allowing subclasses to provide specific implementations, the pattern ensures consistency in the widget structure while allowing for flexibility and customization.


---

## Conclusion

The Template Method Pattern offers a robust solution for defining a consistent structure for a set of algorithms, while allowing customization through subclassing. It promotes code reusability and maintainability, making it a valuable pattern in many front-end development scenarios, including component design, form handling, and lifecycle management.




