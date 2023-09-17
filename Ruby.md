#seed 
upstream: [[Software Development]]

### Introduction

**Ruby** is a high-level, interpreted, and [[Object Oriented Programming (OOP)]] language developed by Yukihiro "Matz" Matsumoto in the mid-1990s. It's known for its easy-to-read syntax that emphasizes simplicity and productivity.

### Getting Started with Ruby

#### Installation 
 [docs](https://www.ruby-lang.org/en/documentation/installation/)
- For Unix/Linux/Mac: You can use a package manager such as `apt`, `yum`, or `brew` to install Ruby.
- For Windows: You can use RubyInstaller.

#### Running Ruby

- You can run Ruby interactively with **IRB (Interactive Ruby Shell)** or directly run `.rb` scripts.

### Basic Syntax

Ruby's basic syntax includes **methods**, **classes**, **blocks**, and **control structures**.

```ruby
puts "Hello, World!" # This is a comment
```

### Variables

Ruby supports **five** types of variables:

1. Local variables (`var`)
2. Instance variables (`@var`)
3. Class variables (`@@var`)
4. Global variables (`$var`)
5. Constants (`CONST`)

### Data Types

Ruby has several built-in data types, including:

- **Numbers**: Integer, Float, Complex
- **Strings**: Sequences of characters
- **Symbols**: Immutable, reusable identifiers (`:symbol`)
- **Arrays**: Ordered collections of objects (`[1, 2, 3]`)
- **Hashes**: Key-value pairs (`{key: "value"}`)
- **Booleans**: `true` or `false`
- `nil`: Represents "nothing" or "no value"

### Control Structures

Control structures in Ruby include `if`-`else`-`end`, `unless`, `case`-`when`-`else`-`end`.

```ruby
if condition
  # do something
elsif other_condition
  # do something else
else
  # do another thing
end
```

### Loops and Iterators

Ruby has several looping constructs, including `while`, `until`, `for`, and the loop method. It also has powerful iterator methods such as `each`, `map`, `select`.

```ruby
5.times do |i|
  puts i
end
```

### Methods

Methods are defined using the `def` keyword.

```ruby
def greet(name)
  puts "Hello, #{name}!"
end

greet("World") # => Hello, World!
```

### Blocks, Procs, and Lambdas

Blocks are anonymous pieces of code that can be passed into methods. Procs and lambdas are types of blocks that can be stored in variables.

```ruby
[1, 2, 3].each do |num|
  puts num * 2
end
```

### Modules and Mixins

Modules are collections of methods and constants. They can be mixed into classes using the `include` method, achieving a "has-a" relationship.

```ruby
module Flyable
  def fly
    puts "I'm flying!"
  end
end

class Bird
  include Flyable
end

bird = Bird.new
bird.fly # => I'm flying!
```

### Error Handling

Ruby uses the `begin`-`rescue`-`end` syntax for exceptions.

```ruby
begin
  # risky operation
rescue SomeExceptionClass => e
  # handle exception
end
```

### File I/O

Ruby has several methods to read or write to files.

```ruby
File.open("file.txt", "r") do |file|
  puts file.read
end
```

### Object-Oriented Programming

Ruby is a pure object-oriented language. It supports inheritance, encapsulation, and polymorphism.

```ruby
class Animal
  def speak
    "I'm an animal."
  end
end

class Cat < Animal
  def speak
    "Meow!"
  end
end

cat = Cat.new
puts cat.speak # => Meow!
```

### Conclusion

Ruby is a versatile and powerful language with a focus on simplicity and productivity. While this guide provides an introduction to its main features, there are many more to explore as you continue your Ruby journey.

### References

- [The Ruby Programming Language](https://www.ruby-lang.org/en/documentation/)
- [Ruby Documentation](https://ruby-doc.org/)
- [Ruby Style Guide](https://rubystyle.guide/)
- [Learn Ruby the Hard Way](https://learnrubythehardway.org/book/)
- [Why's (Poignant) Guide to Ruby](https://poignant.guide/)
- [Ruby Koans](http://rubykoans.com/)