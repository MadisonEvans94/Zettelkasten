#incubator 
upstream: [[Databases]], [[Cloud Computing and Distributed Systems]]
[Redis Explained in 100 seconds](https://www.youtube.com/watch?v=G1rOthIU-uo&ab_channel=Fireship)

### Introduction

![[Pasted image 20230622132724.png]]

**Redis** (*`REmote DIctionary Server`*) is an open-source, in-memory data structure store that can be used as a **database**, **cache**, and **message broker**. It supports various [[Data Structures]] such as **strings**, **hashes**, **lists**, **sets**, and more.

Redis operates in-memory for maximum performance but also provides mechanisms to persist data to **disk**, making it versatile in a wide variety of use cases

### Standard Req/Res flow: 
![[IMG_6FB23D172241-1.jpeg]]

#### Redis contains the requested data
1. Client sends **`GET`** request to application server
2. The application server sends a **`GET`** request to Redis server in order to query for the data
3. Redis **has** the requested data, and so it sends a response with the requested data as payload
4. Application server sends the data back in the response body to the client that originally requested it 

#### Redis does **not** contain the requested data
1. Client sends **`GET`** request to application server
2. Application server sends a **`GET`** request to Redis for the data. Redis **does not have** the data, so it sends a null response back to application server 
3. Next, the application server sends a database query to the database (or a **`GET`** request to a computing instance), and the database responds with the data needed
4. The application server then adds this data as an entry into the Redis store. This step is also known as "[[Caching]]".
5. Finally, the application server delivers the data to the original client within the body of its response 

*The benefit of this setup is that future requests for the same data can be fulfilled quickly and efficiently from the Redis cache, reducing load on the main database and providing faster response times for the client.*
### What's the Difference Between Redis and a normal Key-Value Data structure? 

*Redis is an in-memory data structure store. What about Redis sets it apart from just a standard key-value store? It seems like both have the same definition*

While Redis is often thought of as a **key-value store**, it is much more than that because of its support for a variety of data structures. Here's what sets Redis apart from a standard key-value store:

#### 1. **Data Structure Types:** 

A key-value store typically only stores **strings**, where the key is a string and the value is also a string. 

*On the other hand*, Redis supports a variety of **data structures**, including **[[Strings]]**, **[[Lists]]**, **[[Sets]]**, **[[Sorted Sets]]**, **[[Hashes]]**, **[[Bitmaps]]**, **[[hyperloglogs]]**, and **[[streams]]**. This versatility allows developers to solve a variety of problems in an efficient way.

2. **Operations on Types:** In addition to storing these different data types, Redis also provides a rich set of operations on these types, such as pushing or popping items from a list, adding or removing elements from a set, incrementing the number in a string, etc. This capability lets you execute complex scenarios and transactions directly within the database layer, instead of having to load data into your application, modify it, and then store it back.

3. **Performance:** Since Redis is an in-memory store, it offers high-speed reading and writing of data. While many key-value stores are disk-based, Redis's in-memory nature enables it to perform operations extremely quickly.

4. **Persistence:** Despite being an in-memory store, Redis offers various mechanisms for persisting data to disk. This way, you can enjoy the speed benefits of in-memory operations, without losing your data when the process exits or the machine reboots.

5. **Pub/Sub:** Redis has built-in support for Pub/Sub messaging paradigms, allowing it to be used as a message broker in addition to being a data store.

6. **Transactions:** Redis supports transactions, which means you can group commands together and have them executed sequentially and atomically.

7. **Scripting:** Redis supports Lua scripting, which means you can write scripts to perform complex operations directly on the Redis server.

So while Redis is a key-value store at its core, its support for advanced data structures and operations, as well as other features, make it much more versatile and powerful than a standard key-value store.

### Why Use Redis?

*Here are some reasons why developers might choose to use Redis*

#### Performance: 
Redis is an in-memory data store, which makes it incredibly fast. This speed is useful for **caching**, **real-time analytics**, and similar use cases where speed is crucial.
  
#### Data Structures: 
Redis supports various data types, providing a lot of flexibility in managing data. It can store **strings**, **lists**, **sets**, **sorted sets**, **hashes**, **bitmaps**, and **hyperlog logs**.
  
#### Pub/Sub Capabilities: 
Redis has built-in support for [[pub sub]], allowing it to be used as a **message broker**.
  
#### Persistence: 
While Redis operates in-memory, it also provides options to persist data to disk, combining the speed of in-memory processing with the durability of disk-based storage.
  
#### Atomic Operations: 
Redis operations are **atomic**, which ensures data integrity even in concurrent processing.
  
### Redis Data Types

*Here are the main data types Redis supports:*

#### String: 
This is the simplest data type in Redis, a binary-safe string that can hold any kind of data.

#### Lists: 
These are collections of string elements sorted according to the order they were inserted.

#### Sets: 
Unordered collections of unique strings.

#### Hashes: 
Redis hashes are maps between string fields and string values. 

#### Sorted Sets: 
Similar to sets but every string element is linked to a floating number value, called a score. The elements are sorted by their score.
  
### How to Use Redis

Redis can be installed and run on your local machine or a server. After installing and starting the Redis server, you can interact with it using the **`redis-cli`** (command-line interface).

*Here is a simple example:*

```bash
$ redis-cli
127.0.0.1:6379> SET name "Redis"
OK
127.0.0.1:6379> GET name
"Redis"
```

In this example, we used the `SET` command to set a key `name` with a value `Redis`, then used the `GET` command to retrieve the value of the key `name`.

### Redis Commands

*Here are some of the commonly used Redis commands:*

#### `SET key value`:
Set the string value of a key.

#### `GET key`: 
Get the value of a key.

#### `DEL key`: 
Delete a key.

#### `EXISTS key`: 
Determine if a key exists.

#### `EXPIRE key seconds`: 
Set a key's time to live in seconds.

#### `LPUSH key value`: 
Prepend one or multiple values to a list.

#### `LPOP key`: 
Remove and get the first element in a list.

#### `SADD key member`: 
Add one or more members to a set.

#### `SISMEMBER key member`: 
Determine if a given value is a member of a set.

#### `PUBLISH channel message`:
Post a message to a channel.

### Redis with Node.js

You can also interact with Redis programmatically through various language-specific clients. Let's take a look at using Redis with Node.js.

*First, you need to install the `redis` npm package:*

```bash
npm install redis
```

*Here is a simple example of using `Redis` in `Node.js`:*

```javascript
const redis = require('redis');
const client = redis.createClient();

client.on('connect', function() {
    console.log('Connected to Redis...');
});

client.set('name', 'Redis', function(err, reply) {
    console.log(reply);  // prints "OK"
});

client.get('name', function(err, reply) {
    console.log(reply);  // prints "Redis"
});
```

In this example, we connect to the Redis server using `redis.createClient()`. We then use `client.set()` to set a key and `client.get()` to retrieve the key value.

### Redis as a Multi-Model DB 

![[Screen Shot 2023-06-22 at 2.35.09 PM.png]]
Redis can be modeled as a database also, and here's just a *few* ways how 

#### For Search Queries...

Use [[Redis Search]] to turn DB into full text search engine

![[Screen Shot 2023-06-22 at 2.38.43 PM.png]]

#### If your data is hierarchal...
Use [[Redis JSON]] for a document oriented database 


![[Screen Shot 2023-06-22 at 2.37.46 PM.png]]


#### If your data contains relationships...
use [[Redis Graph]] and query it with [[Cypher Query Language]] 

![[Screen Shot 2023-06-22 at 2.36.34 PM.png]]

## Conclusion

Redis is a powerful tool that offers high performance, flexible data types, and various features. It's an excellent choice for applications that need a fast, flexible in-memory data store.

Remember, as with any technology, it's important to understand the capabilities and use cases of Redis to use it effectively in your applications.