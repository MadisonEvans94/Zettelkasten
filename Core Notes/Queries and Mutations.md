#seed 
upstream: [[GraphQL]]

---

**links**: [documenation](https://graphql.org/learn/queries/)

---

Brain Dump: 

--- 


## Introduction 

On this page, you'll learn in detail about how to query a GraphQL server.

## Fields 

At its simplest, GraphQL is about asking for specific fields on objects. Let's start by looking at a very simple query and the result we get when we run it:

>query
```graphql
{
  hero {
	# Queries can have comments!
    name
    appearsIn
  }
}
```

>response 
```json
{
  "data": {
    "hero": {
      "name": "R2-D2"
    }
  }
}
```

You can see immediately that the query has exactly the same shape as the result. This is essential to GraphQL, because you always get back what you expect, and the server knows exactly what fields the client is asking for.

The field `name` returns a `String` type, in this case the name of the main hero of Star Wars, `"R2-D2"`.

In the previous example, we just asked for the name of our hero which returned a String, but fields can also refer to Objects. In that case, you can make a _sub-selection_ of fields for that object. GraphQL queries can traverse related objects and their fields, letting clients fetch lots of related data in one request, instead of making several roundtrips as one would need in a classic REST architecture.

>query
```
{
  hero {
    name
    friends {
      name
    }
  }
}
```

>response
```json
{
  "data": {
    "hero": {
      "name": "R2-D2",
      "friends": [
        {
          "name": "Luke Skywalker"
        },
        {
          "name": "Han Solo"
        },
        {
          "name": "Leia Organa"
        }
      ]
    }
  }
}
```

Note that in this example, the `friends` field returns an array of items. GraphQL queries look the same for both single items or lists of items; however, we know which one to expect based on what is indicated in the schema.

## Arguments 

If the only thing we could do was traverse objects and their fields, GraphQL would already be a very useful language for data fetching. But when you add the ability to pass arguments to fields, things get much more interesting.

>query
```
{
  human(id: "1000") {
    name
    height
  }
}
```

>response
```json
{
  "data": {
    "human": {
      "name": "Luke Skywalker",
      "height": 1.72
    }
  }
}
```

In a system like REST, you can only pass a single set of arguments - the query parameters and URL segments in your request. But in GraphQL, every field and nested object can get its own set of arguments, making GraphQL a complete replacement for making multiple API fetches. You can even pass arguments into scalar fields, to implement data transformations once on the server, instead of on every client separately.

Arguments can be of many different types. GraphQL comes with a default set of types, but a GraphQL server can also declare its own custom types, as long as they can be serialized into your transport format.

## Aliases

If you have a sharp eye, you may have noticed that, since the result object fields match the name of the field in the query but don't include arguments, you can't directly query for the same field with different arguments. That's why you need _aliases_ - they let you rename the result of a field to anything you want.

>query
```
{
  empireHero: hero(episode: EMPIRE) {
    name
  }
  jediHero: hero(episode: JEDI) {
    name
  }
}
```

>response
```json
{
  "data": {
    "empireHero": {
      "name": "Luke Skywalker"
    },
    "jediHero": {
      "name": "R2-D2"
    }
  }
}
```

In the above example, the two `hero` fields would have conflicted, but since we can alias them to different names, we can get both results in one request.

