#seed 
upstream: [[Dart]]

---

**video links**: 

---

# Brain Dump: 


--- 


Generics are a powerful feature in Dart that allows you to write flexible, reusable, and type-safe code. They are widely used in Flutter to enhance the maintainability and robustness of the codebase. This document provides an in-depth look at generics, their usage, and examples.

## Introduction to Generics

Generics enable you to write code that works with different types without sacrificing type safety. They can be used with classes, methods, and functions.

### Why Use Generics?

1. **Type Safety**: Generics catch potential type errors at compile time rather than at runtime.
2. **Code Reusability**: Write a single class or function that can work with different types.
3. **Improved Readability**: Clearly define what types a class or function is expected to work with.

## Generic Types

### Using Generics with Collections

Dart collections like `List`, `Set`, and `Map` support generics to define the types they contain.

```dart
List<String> names = ['Alice', 'Bob'];
Map<String, int> ages = {'Alice': 30, 'Bob': 25};
```

### Custom Classes with Generics

You can define custom classes that use generics. Here's an example:

```dart
class Box<T> {
  final T value;

  Box(this.value);
}

Box<int> intBox = Box(10);
Box<String> stringBox = Box('Hello');
```

## Generic Functions and Methods

Generics can also be used with functions and methods.

```dart
T first<T>(List<T> items) {
  return items[0];
}

int firstInt = first<int>([1, 2, 3]);
String firstString = first<String>(['A', 'B', 'C']);
```

## Bounded Types

You can restrict the types that can be used with generics by using bounds.

```dart
class Animal {}
class Dog extends Animal {}

class Kennel<T extends Animal> {
  final List<T> animals = [];
}

Kennel<Dog> dogKennel = Kennel();
```

## Conclusion

Generics are an essential part of Dart and Flutter development, providing a means to write more flexible and safe code. By understanding how to use them with classes, methods, and functions, you can create more maintainable and robust applications.

Understanding generics helps you work with Flutter's many APIs that leverage them, such as working with `Future<T>`, collections, and custom widgets.

---

This document should cover the essentials of generics in Dart and Flutter. It includes the reasoning behind using generics, examples of using generics with collections, defining generic classes and functions, and bounded types. Feel free to refer back to this guide as you continue developing with Flutter.




