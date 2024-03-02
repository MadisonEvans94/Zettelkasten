
#seed 
upstream: [[Django]]

---

**links**: 

---

Brain Dump: 

--- 







## What are Django Signals?

Django signals are a form of Observer Pattern, a design pattern used in software engineering. They allow decoupled applications to receive notifications when certain actions occur. In Django, signals are used to allow different parts of an application to communicate with each other.

## `post_save` Signal

The `post_save` signal is sent by Django's model system when a model instance is saved. This signal can be used to perform actions after a model instance is saved to the database.

## Use Cases

### 1. **Updating Related Data**

One common use case for `post_save` signals is to update or modify related data. For instance, in a family tree application, when a `FamilyMember` instance is saved, you might want to automatically update related fields in other instances, like adding a child to a parent's children list.

### 2. **Two-Way Relationships**

In situations where models have two-way relationships (like spouses in a family tree), a `post_save` signal can be used to ensure that both sides of the relationship are updated. When one instance is updated to reference another, the other instance is automatically updated to reference the first.

### 3. **Data Integrity and Business Logic**

`post_save` signals can enforce business logic or data integrity rules that go beyond what's possible with standard Django model fields and validators.

## Best Practices

### 1. **Avoid Recursive Calls**

When using `post_save` signals to modify related model instances, care must be taken to avoid recursive calls which can lead to infinite loops.

### 2. **Use Signals Judiciously**

Signals can make debugging more challenging and can lead to tightly coupled code if overused. It's often better to use more explicit ways of handling related updates, such as overriding the `save` method, unless you specifically need the decoupling that signals provide.

### 3. **Test Thoroughly**

Thoroughly test the functionality that relies on signals to ensure it behaves as expected and does not have unintended side effects.

---

This document provides an overview of Django signals and `post_save`, highlighting their common use cases and best practices, particularly in the context of updating related data and handling complex relationships in models.