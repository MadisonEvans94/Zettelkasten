#seed 
upstream: [[Computer Networks]]

---

**links**: 

---

Brain Dump: 

--- 

## What are Sockets? 

Sockets are a fundamental part of the internet architecture. They are an abstraction used in network programming to provide a way to send and receive data over a network. Essentially, a socket is an endpoint for communication between two machines. Here's a more detailed explanation:

1. **Endpoint for Communication**: A socket serves as an endpoint for sending and receiving data across a computer network. It is based on the client-server model of communication.

2. **IP Address and Port Number**: Each socket is identified by an IP address and a port number. The IP address indicates the host, and the port number specifies the application or service within the host. This combination allows multiple network applications to run simultaneously on a single machine without interference.

3. **Types of Sockets**:
    - **Stream Sockets (TCP Sockets)**: These provide a reliable, connection-oriented service. They ensure that data arrives sequentially and without errors (using the Transmission Control Protocol, TCP).
    - **Datagram Sockets (UDP Sockets)**: These provide a connectionless service for sending individual packets of data. They use the User Datagram Protocol (UDP) and do not guarantee delivery or order, but they are faster and more efficient for certain purposes.

4. **Socket APIs**: Most operating systems provide a Socket API (Application Programming Interface) that allows programmers to create, manage, and use sockets in network software. This API includes functions for creating a socket, connecting to a server, sending and receiving data, and closing the connection.

5. **Role in Client-Server Model**: In typical client-server applications, the server listens on a specific port. When a client wants to communicate with the server, it creates a socket and requests a connection. If the server accepts the connection, a new socket is created for the server-client communication.

6. **Use in Various Protocols**: Sockets are used in various network protocols, not just TCP and UDP. They are a key component in enabling network communication for applications like web browsers, email clients, and many other internet services.

Sockets provide a way to abstract the complexities of network communication, allowing developers to focus on the specifics of their application rather than the intricacies of network protocols and data transmission.


