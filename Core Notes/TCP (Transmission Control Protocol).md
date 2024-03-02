#seed 
upstream: [[Network Engineering]], [[cyber security]]
[[Network Address Translation (NAT)]]
## TCP: Transmission Control Protocol
#TODO: understand difference between stateless and stateful firewalls

Transmission Control Protocol (TCP) is a standard that defines how to establish and maintain a network conversation through which application programs can exchange data. TCP works with the Internet Protocol (IP), which defines how computers send packets of data to each other.


### Overview

TCP is one of the main protocols in TCP/IP networks. Whereas the IP protocol deals only with packets, TCP enables two hosts to establish a connection and exchange streams of data. TCP guarantees delivery of data and also guarantees that packets will be delivered in the same order in which they were sent.

### Key Features of TCP

#### 1. Connection-Oriented

TCP is a connection-oriented protocol, which means it requires handshaking to establish end-to-end communications. Once a connection is set up, user data can be sent bidirectionally over the connection.

#### 2. Reliable

TCP is reliable as it guarantees delivery of data to the destination router. The delivery of data to the destination cannot be guaranteed in an IP network. TCP adds support to detect errors or lost data and to trigger retransmission until the data is correctly and completely received.

#### 3. Ordered

TCP rearranges data packets in the order specified. When packets arrive in the wrong order, TCP buffers delay the out-of-order data until all data can be properly re-ordered and delivered to the application.

#### 4. Heavyweight

TCP is heavyweight. TCP requires three packets to set up a socket connection, before any user data can be sent. TCP handles reliability and congestion control.

### TCP Segment Structure

![[TCP Segment Structure.png]]

A TCP segment consists of a segment header and a data section. The TCP header contains 10 mandatory fields, and an optional extension field (Options):

#TODO: add descriptions for what each portion of the header is for 

- Source port (16 bits)
- Destination port (16 bits)
- Sequence number (32 bits)
- Acknowledgment number (32 bits)
- Data offset (4 bits)
- Reserved (3 bits)
- Flags (9 bits - NS, CWR, ECE, URG, ACK, PSH, RST, SYN, FIN)
- Window size (16 bits)
- Checksum (16 bits)
- Urgent pointer (16 bits)
- Options (Variable 0-320 bits, divisible by 32)

### Three-way Handshake in TCP

The TCP three-way handshake in Transmission Control Protocol (also called the TCP-handshake; three message handshake and/or SYN-SYN-ACK) is the method used by TCP set up a TCP/IP connection over an Internet Protocol based network.

![[Pasted image 20230623160928.png]]

**SYN:** The active open is performed by the client sending a SYN to the server. The client sets the segment's sequence number to a random value A.

**SYN-ACK:** In response, the server replies with a SYN-ACK. The acknowledgment number is set to one more than the received sequence number i.e. A+1, and the sequence number that the server chooses for the packet is another random number, B.

**ACK:** Finally, the client sends an ACK back to the server. The sequence number is set to the received acknowledgement value i.e. A+1, and the acknowledgement number is set to one more than the received sequence number i.e. B+1.

### TCP Congestion Control

TCP uses a number of mechanisms to achieve high performance and avoid congestion collapse, where network performance can fall by several orders of magnitude. These mechanisms control the rate of data entering the network, keeping the data flow below a rate that would trigger collapse.

They include:

1. **Slow-start:** It is a mechanism used by the sender to avoid sending more data than the network is capable of transmitting, that is, to avoid network congestion. It is used in conjunction with other algorithms to control congestion.

2. **Congestion Avoidance:** After the slow-start phase ends, TCP continues sending more packets on the network, but it increases the number of packets linearly.

3. **Fast Retransmit:** It is an enhancement to TCP retransmission that retransmits a lost packet before the sender's timeout expired.

4. **Fast Recovery:** It is an algorithm that reduces the time a connection is in the recovery state.

### Summary

TCP is a protocol that ensures reliability in a transmission, which ensures that there is no loss of packets, that the packets are in the right order, that the delay is acceptable, etc. Understanding the functionality of TCP is essential for any network administrator, cyber security professional or software developer. It is an intricate part of all web browsing and content delivery.
