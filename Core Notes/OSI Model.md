#evergreen1 
upstream:

---

**links**: https://www.youtube.com/watch?v=eKHCH6rw0As&ab_channel=_DrunkEngineer_

---

Brain Dump: 
- [ ] send images from iphone 
[[Encapsulation Analogy]]
[[Address Resolution Protocol]]
[[Route Tables]]
[[Decimal Binary Conversions]]

--- 


## Overview 
![[Osi model.pdf]]

The **OSI** model, which stands for Open Systems Interconnection model, is a conceptual framework used to understand and implement standards for network communication. The image above illustrates how a piece of data, like an HTTP request, is transformed as it travels through the layers of the OSI model.

The image shows an encapsulation process where each layer adds its own header (and sometimes a trailer) to the data unit from the layer above. This process is crucial because each layer provides specific functionalities necessary for successful communication. The headers (and trailers) contain control information that the corresponding layer on the receiving end needs to understand to process and eventually de-encapsulate the message back to its original form.

When the data arrives at its destination, each layer will strip off its respective header (and trailer if applicable), interpreting the control information and acting accordingly until the data is back in its original form and passed up to the destination application.

This layering provides modularization of network functionality which makes design, development, troubleshooting, and learning about networks (like in your course) much more manageable.

---

## The Layers

### Layer 7: Application

The application layer includes multiple protocols, some of the most popular ones include: 1) The HTTP protocol (web), SMTP (e‐mail), 2) The FTP protocol (transfers files between two end hosts), and 3) The DNS protocol (translates domain names to IP addresses). So the services that this layer offers are multiple depending on the application that is implemented.

- **Data Unit**: Application Data
- **Function**: This is where network processes to applications occur, such as an HTTP GET request. Here, the user’s data is prepared for transport. This layer handles issues like network transparency, resource allocation, and problem partitioning.
- **What's Appended**: The HTTP headers are added here, including the request type (GET), the destination (like an IP address and port), and other headers like cookies and content types.

#### Example 

Layer 7, the Application Layer in the OSI model, is where network services interact directly with end-user applications. It deals with the actual payload of the communication — the data the user wants to send or receive.

A data unit at Layer 7 might be:

##### For an HTTP GET Request
```http
GET /index.html HTTP/1.1
Host: www.example.com
User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3
Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8
Accept-Language: en-US,en;q=0.5
Accept-Encoding: gzip, deflate
Connection: keep-alive
Cookie: userID=12345; sessionToken=abcde12345xyz
```

In this example:
- The first line is the request line, which includes the method (`GET`), the requested resource (`/index.html`), and the HTTP version (`HTTP/1.1`).
- Following the request line are various headers that provide additional information about the request, such as `Host` (the domain name of the server), `User-Agent` (information about the client making the request), and `Accept` (the types of content the client can process).
- `Cookie` is another header included here, which is a piece of data stored on the user's computer by the web browser while browsing that website. The server reads this data to recall the user's previous activity.

##### For an Email (SMTP Protocol)
```smtp
EHLO mail.example.com
MAIL FROM:<user@example.com>
RCPT TO:<friend@example.net>
DATA
Subject: Hello

Hi there,

This is an example email!

Best,
User
.
QUIT
```

In this SMTP transaction:
- `EHLO` starts the conversation with the server, identifying the sender's domain.
- `MAIL FROM` and `RCPT TO` specify the sender and recipient email addresses.
- `DATA` indicates that the following lines constitute the body of the message, starting with headers (`Subject:`) and then the body of the email.
- The period on a line by itself (`.`) signals the end of the message body.
- `QUIT` ends the SMTP session.

Each of these represents what the Application Layer might be handling in terms of data and is not concerned with how the data is transmitted over the network. The Application Layer is about the content and syntax of the information exchange. The actual bits sent over the wire (or wireless) include much more than these examples, as additional information is added by each subsequent layer.

### Layer 6: Presentation

The presentation layer plays the intermediate role of formatting the information that it receives from the layer below and delivering it to the application layer. For example, some functionalities of this layer are formatting a video stream or translating integers from big endian to little endian format.

- **Data Unit**: Still considered Application Data or Presentation Data
- **Function**: This layer translates the application format to network format and vice versa. It handles data encryption, compression, and translation between different data formats.
- **What's Appended**: If necessary, encryption is added at this layer to secure the data.

#### Example 

Layer 6 of the OSI model is the Presentation Layer. Its primary function is to translate, encrypt, and compress data for the application layer. In practical networking, this layer is not always explicitly implemented as a separate layer, because its functions can be part of the application or part of the protocols used in the application layer.

Here's what might happen at Layer 6
##### Translation
Data from the application layer is put into a format that can be sent over the network. This could involve changing character encoding formats (such as from ASCII to EBCDIC).

Before: `Hello, World!` (in ASCII)
After: `C1C8C5D3D6D2D940D6D3D4C4!` (in EBCDIC)

##### Encryption
Data can be encrypted for security purposes. If an application requests encrypted transmission, the presentation layer encrypts the data before it's passed down to the session layer.

Before: `Hello, World!`
After: `3a0c8eaaebbf...` (an encrypted string of bytes)
##### Compression
Data can be compressed to reduce its size for transmission. This is especially useful for reducing bandwidth usage and improving transmission speed.

Before: A long text document.
After: A compressed binary blob, which is smaller in size (represented by a binary or hex sequence).

In the context of a specific protocol that performs these functions, consider the Secure Sockets Layer (SSL) or its successor, Transport Layer Security (TLS). These protocols work at the presentation layer to encrypt data sent from the application layer.

In most networking stacks in use today (like the TCP/IP stack), you won't find a separate presentation layer. Instead, applications handle these functions directly (e.g., a web browser using HTTPS will handle the encryption and compression internally).
### Layer 5: Session

The session layer is responsible for the mechanism that manages the different transport streams that belong to the same session between end‐user application processes. For example, in the case of tele‐ conference application, it is responsible to tie together the audio stream and the video stream.

- **Data Unit**: Session Protocol Data Unit (SPDU)
- **Function**: This layer establishes, manages, and terminates sessions between two communicating hosts. It also synchronizes dialogue between the two hosts' presentation layers and manages their data exchange.
- **What's Appended**: A session tag is appended to keep track of the dialogue control.

#### Example 

An example of a simple session tag could be a string like this:

`Session-ID: 12345; Seq: 678; Token: A1B2-C3D4`

- `Session-ID` is the unique identifier for the session.
- `Seq` is a sequence number indicating the position of this particular data packet in the sequence of the entire session.
- `Token` is a dialogue token that might be used to manage control of the communication.

>In practical implementations within actual network protocols, such as TCP/IP, the functionality of the OSI session layer is often rolled into the transport layer. For instance, in TCP, there is no explicit session layer, but TCP headers include a sequence number which serves a similar purpose to the session tag in tracking the order of segments.

### Layer 4: Transport
#### Multiplexing: 

In networking, multiplexing refers to a host's ability to manage multiple connections simultaneously by using different ports for each connection. This allows several applications to operate concurrently. Data is encapsulated into **segments**, with headers specifying the sending and receiving ports. Demultiplexing is the process of receiving these segments and directing the data to the appropriate port based on the header information.

In connection-oriented multiplexing, such as with TCP (Transmission Control Protocol), each connection is established distinctly before data transfer begins. This ensures a reliable communication path, where segments are associated with a specific connection identified by a source and destination port pair, along with the IP addresses of the sender and receiver. The protocol ensures that data is delivered in order and without loss.

In contrast, connectionless multiplexing, as used in UDP (User Datagram Protocol), does not establish a dedicated connection before sending data. Here, multiplexing involves sending and receiving data packets (datagrams) independently, each identified only by the source and destination port numbers in the header. Demultiplexing in this scenario involves directing these datagrams to the correct application based on the port number, without the assurance of delivery order or error correction.

#### 3 Way Handshake 

The TCP three-way handshake is a process used to establish a TCP connection between a client and a server. It involves three steps:

1. **SYN**: The client initiates the connection by sending a segment with a SYN (synchronize) flag set. This segment includes an initial sequence number (ISN), which is a random number chosen by the client to begin the sequence numbering of the bytes it will send.

2. **SYN-ACK**: Upon receiving the SYN request, the server responds with a segment that has both SYN and ACK (acknowledgment) flags set. The ACK number in this segment is the client's initial sequence number plus one, acknowledging the receipt of the client's SYN segment. The server also includes its own initial sequence number for the bytes it will send.

3. **ACK**: The client then sends an ACK segment to the server. The ACK number in this segment is the server's initial sequence number plus one. This final step completes the three-way handshake, and the connection is established.

After the handshake, data transfer can begin. The sequence number (not 'syn value') for each subsequent segment sent in either direction is incremented by the number of bytes sent in the previous segment, ensuring proper ordering and reliable delivery. The connection remains open until it is explicitly closed by a TCP termination process.

- [ ] transmission control 
- [ ] Flow control 
- [ ] congestion control 
	- [ ] what are the goals of transmission control 
	- [ ] congestion control flavors: E2E vs network assisted 
	- [ ] how a host infers congestion/signs of congestion 
- [ ] How does a TCP sender limit the sending rate 
- [ ] congestion control @ TCP 
- [ ] slow start in tcp 
- [ ] tcp fairness 
- [ ] tcp throughput 

The transport layer is responsible for the end‐to‐end communication between end hosts. In this layer, there are two transport protocols, namely TCP and UDP. The services that TCP offers include: a connection‐oriented service to the applications that are running on the layer above, guaranteed delivery of the application‐layer messages, flow control which in a nutshell matches the sender’s and receiver’s speed, and a congestion‐control mechanism, so that the sender slows its transmission rate when it perceives the network to be congested. On the other hand, the UDP protocol provides a connectionless best‐effort service to the applications that are running in the layer above, without reliability, flow or congestion control. At the transport layer, we refer to the packet of information as a segment.
[[TCP (Transmission Control Protocol)]]
- **Data Unit**: Segment or Datagram (for TCP or UDP respectively)
- **Function**: This layer ensures that messages are delivered error-free, in sequence, and with no losses or duplications. It does this by segmenting data and managing each piece's reassembly at the destination.
- **What's Appended**: Transport layer header is added which includes port numbers for communication, sequence numbers for reassembly, and error-checking information.

#### Example 
Layer 4 of the OSI model is the Transport Layer, which is critical for managing the transmission of data between systems and hosts. Its responsibilities include segmenting data from the upper layers, managing each segment's delivery, and ensuring the reliability of the communication.

In the TCP/IP stack, this layer is where the Transmission Control Protocol (TCP) and User Datagram Protocol (UDP) operate.

##### TCP:


Suppose you're sending an email, which is a relatively large amount of data. The Transport Layer would break this email into smaller segments. Each segment would be given a sequence number to keep track of the order.

Here's what a segment might look like in a very simplified form:

- **Segment Header** (TCP header might include):
  - Source port: `12345`
  - Destination port: `25` (for SMTP)
  - Sequence number: `1001`
  - Acknowledgment number (if part of an established session): `5678`
  - Flags (SYN, ACK, FIN, etc.): `ACK` (acknowledgment of the received segment)
  - Window size (flow control): `5000` (number of bytes sender is willing to receive)

- **Segment Data**: [Part of the email text]

- **Error Checking**: A checksum is calculated for the segment and included in the header.

##### UDP:
Let's say you're streaming video. UDP, which is connectionless and doesn't require acknowledgment of receipt, sends out packets (called datagrams) of the video data. A datagram might look like this:

- **Datagram Header** (UDP header):
  - Source port: `54321`
  - Destination port: `80` (assuming HTTP over UDP, which is uncommon)
  - Length: `1000` (length of the UDP header and data)
  - Checksum: (for error checking)

- **Datagram Data**: [Binary data of the video stream]

The Transport Layer's role is vital in managing the data transport reliably (with TCP) or quickly (with UDP), depending on the needs of the application.

### Layer 3: Network
- **Data Unit**: Packet
- **Function**: This layer handles data packet routing through the network, including logical addressing and path determination.
- **What's Appended**: The network header is appended, which includes the logical addresses of the sender and receiver (IP addresses).

![[Layer 3 Summary.png]]

#### Example 

The data unit at this layer is called a packet. Here's what a simplified IP packet might contain:

- **Packet Header**:
    
    - Source IP address: `192.168.1.2`
    - Destination IP address: `93.184.216.34`
    - Time-to-Live (TTL): `64`
    - Protocol (identifies the next level protocol, e.g., TCP or UDP): `6` for TCP
- **Packet Payload**: This is the segment from Layer 4, including the Layer 4 header and actual data.
    
- **Routing Information**: Though not part of the packet header per se, the routing process may involve using algorithms and routing tables to determine the next hop and the best path for the packet.
### Layer 2: Data Link
- **Data Unit**: Frame
- **Function**: This layer is responsible for node-to-node data transfer and for detecting and possibly correcting errors that may occur in the physical layer.
- **What's Appended**: A frame header and trailer are added. The header includes the hardware addresses (MAC addresses) of the sender and receiver, while the trailer typically contains a frame check sequence for error detection.
#### Example 

The data unit at this layer is called a frame. A frame encapsulates a packet and adds header and trailer information necessary for that link level communication.

- **Frame Header**:
    
    - Source MAC address: `00-14-22-01-23-45`
    - Destination MAC address: `00-67-89-01-23-45`
    - Ethertype (to indicate which protocol is encapsulated in the payload of the frame): `0x0800` for an IPv4 packet
- **Frame Payload**: This includes the entire packet from Layer 3.
    
- **Frame Trailer**:
    
    - Frame Check Sequence (FCS): Typically a CRC32 checksum used to detect errors in transmission.

When a frame is received, the node checks the destination MAC address to see if it matches its own (or is a broadcast or multicast it's interested in), and if it does, it will process the frame, check for errors using the FCS, and if it's error-free, pass the payload up to Layer 3.When a frame is received, the node checks the destination MAC address to see if it matches its own

### Layer 1: Physical
- **Data Unit**: Bits
- **Function**: This is the level of the actual hardware. It transmits raw bits over the physical medium and deals with the mechanical and electrical specifications of the data and the transmission medium.
- **What's Appended**: Data is converted into electrical impulses, light or radio signals, etc., that can be transmitted over the appropriate medium.

#### Example 

```binary 
0101000101010110011101010101010101
```


