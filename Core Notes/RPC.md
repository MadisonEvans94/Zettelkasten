#seed 
upstream: [[distributed computing]]

---

**links**: 

---

Brain Dump: 

--- 






RPC, or Remote Procedure Call, is a protocol used in distributed computing systems. It allows a computer program to cause a procedure (subroutine) to execute in another address space (commonly on another computer on a shared network), which is coded as if it were a normal (local) procedure call, without the programmer explicitly coding the details for the remote interaction.

Here are the key aspects of RPC in distributed computing:

1. **Abstraction of Remote Interaction**: RPC abstracts the complexity of the network by allowing the developer to think in terms of calling a function or a method directly, rather than dealing with network protocols and data transmission. This makes the process of interacting with remote services more straightforward and similar to local function calls.

2. **Client-Server Model**: Typically, RPC is used in a client-server model where the client initiates a request for a procedure to be executed on the server. The server then processes the request, executes the procedure, and sends back the result.

3. **Interface Definition Language (IDL)**: RPC systems often use an Interface Definition Language to define the interfaces that can be called remotely. The IDL specifies the data types and the signatures of the remote procedures.

4. **Marshalling and Unmarshalling**: RPC involves converting (marshalling) the procedure arguments into a form that can be transmitted over a network, and then converting (unmarshalling) them back into their original form at the receiver’s end.

5. **Synchronous or Asynchronous Calls**: RPC calls can be synchronous, where the client blocks and waits for the server to finish the procedure and return the result, or asynchronous, where the client continues its execution without waiting and a callback mechanism is used to handle the result.

6. **Error Handling and Exceptions**: Since RPC involves network communication, it must handle errors like network failures, server unavailability, or transmission errors. Exception handling mechanisms are often built into RPC frameworks to manage such scenarios.

7. **Security**: Security is a crucial aspect of RPC, as data is transmitted across a network. Encryption and authentication mechanisms are usually employed to protect data integrity and confidentiality.

8. **Examples**: Popular RPC frameworks and protocols include XML-RPC, JSON-RPC, gRPC (developed by Google), and Apache Thrift.

In summary, RPC makes distributed computing more accessible and manageable by providing a framework for executing functions or procedures on remote systems as if they were local calls.

---

Yes, your understanding is essentially correct. A Web API (Application Programming Interface) can be seen as a more specific implementation of the RPC (Remote Procedure Call) paradigm, particularly in the context of web services. Both concepts are centered around communication between different systems, but they have some distinct characteristics:

1. **RPC in General**: RPC is a broader concept in distributed computing, where a procedure is executed in a different address space (typically on another machine). The key idea is that it allows a function call to be made on one computer (the client) but executed on another (the server), with the complexity of the network communication abstracted away. RPC can be implemented using various protocols and data formats.

2. **Web API as a Form of RPC**: A Web API is a specific type of RPC that operates over the web (HTTP/HTTPS protocols). It is used to enable interaction between different software applications, especially those that run on different machines or are built with different technologies. Web APIs often use HTTP methods (GET, POST, PUT, DELETE) and typically exchange data in formats like JSON or XML.

3. **Call and Response Mechanism**: Both RPC and Web APIs indeed follow a call-and-response mechanism. In RPC, a client sends a request to invoke a procedure on a server, and the server responds with the result of that procedure. Similarly, in a Web API, the client makes a request to a specific URL (representing an API endpoint) and receives a response, usually containing data or confirmation of the request's processing.

4. **Use Cases and Context**: While RPC can be used in a variety of distributed computing contexts (including internal systems, microservices, etc.), Web APIs are particularly common in web development and for building interfaces between different web-based services.

5. **Formats and Protocols**: RPC can use different communication protocols and data formats (like JSON-RPC, XML-RPC, gRPC, etc.), while Web APIs are typically bound to the HTTP/HTTPS protocols and web-friendly data formats like JSON or XML.

So, while Web APIs are a specific implementation of the RPC concept tailored for web-based communication, the fundamental principle of a remote call and response is a common thread that runs through both.