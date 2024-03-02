#evergreen1
upstream: [[Stateless vs Stateful systems]], [[Computer Networks]]

---

**links**: 

---
Certainly! I've expanded and refined your notes on Web Sockets to make them more comprehensive and detailed.

---

## What Is a Web Socket?

A **Web Socket** is a persistent connection between a client and a server that allows for full-duplex communication, meaning data can be sent and received simultaneously. Unlike traditional HTTP connections that are initiated by the client and closed after a response is received, a Web Socket remains *open*, providing a constant connection. This is achieved through a Web Socket handshake initiated over HTTP, where the client requests an upgrade to Web Sockets and the server, if it supports Web Sockets, acknowledges this request. Once this upgrade is complete, the protocol changes from HTTP to Web Sockets (*ws://* or *wss://* for secure connections), allowing for data to flow freely in both directions without the need to establish new connections.

## What Problem Does It Solve

Consider the example of building a chat application, similar to those used during live streams on platforms like Twitch. With traditional HTTP, whenever a user sends a message, it would be done through an HTTP request. To display the most current messages to all users, the server would have to be polled periodically for updates. This polling involves making a new TCP connection every few seconds, which is resource-intensive and inefficient due to the overhead of establishing connections and the latency introduced.

Web Sockets address this issue by establishing a persistent, full-duplex connection between the client and the server. Once the Web Socket handshake is completed, this connection allows for real-time, bidirectional communication. Messages sent by any user can be immediately *pushed* to all connected clients by the server, eliminating the need for polling and significantly reducing the resources and time required to maintain live, interactive communication.

## How Does Data Move Bidirectionally?

In a Web Socket connection, data transmission is **bidirectional** and can occur simultaneously due to the connection being **full-duplex**. This is made possible by the underlying TCP/IP protocol, which Web Sockets use to maintain a persistent connection. Once the handshake is completed, both the client and server can send data frames to each other at any time. These frames can carry text, binary data, or control information for managing the connection.

The protocol defines a frame format that encapsulates data sent over the connection. Each frame has a header that specifies its type (e.g., continuation, text, binary, close, ping, or pong) and its length. The client and server can send these frames independently of each other, allowing for continuous data flow without the need for requesting or waiting for a response, as in traditional HTTP communication. This mechanism ensures that messages or updates can be instantly transmitted and received, facilitating real-time applications like chat services, live feeds, and interactive games.

## What About Streaming in HTTP2?

HTTP/2 introduces several improvements over HTTP/1.x, with a focus on performance optimization and efficient use of resources. One of the notable features of HTTP/2 is its support for multiplexing, which allows multiple requests and responses to be in flight simultaneously over a single TCP connection. This reduces the overhead associated with setting up multiple connections and improves the efficiency of data transmission.

Streaming in HTTP/2 can be seen as an enhancement to the capabilities provided by Web Sockets, although it serves slightly different use cases. HTTP/2's server push feature allows the server to send multiple responses for a single client request. This can be particularly useful for scenarios where the server knows in advance which resources the client will need, enabling the server to push these resources to the client without waiting for individual requests.

While HTTP/2 does not fully replace Web Sockets, as they are designed for different purposes, it offers an alternative for applications that require efficient, bidirectional communication over a single, long-lived connection. HTTP/2 is particularly advantageous for web applications that need to load multiple resources efficiently, while Web Sockets remain the preferred choice for real-time, interactive applications that require low latency communication.

## Example with React and Express 

> see [[Creating a Chat Application With Socket.io]]