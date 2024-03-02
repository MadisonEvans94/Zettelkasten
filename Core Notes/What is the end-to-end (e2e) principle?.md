#incubator
upstream:

---

**links**: 

---

Brain Dump: 

--- 

## Definition 

The **end-to-end (e2e) principle** is a design choice that characterized and shaped the current architecture of the Internet. The e2e principle suggests that specific application-level functions usually cannot, and preferably should not, be built into the lower levels of the system at the core of the network. 

>It's kinda like separation of concerns. Essentially, we're leaving the responsibilities of networking and routing to the bottom 4 layers and all the encryption and application stuff will be handled by the end devices (client and server)

In simple terms, the e2e principle is summarized as follows: the network core should be simple and minimal, while the end systems should carry the intelligence. As mentioned in the seminal paper “End-to-End Arguments in System Design” by Saltzer, Reed, and Clark: 

> "The function in question can completely and correctly be implemented only with the knowledge and help of the application standing at the endpoints of the communications system. Therefore, providing that questioned function as a feature of the communications systems itself is not possible.”

The same paper reasoned that many functions can only be completely implemented at the endpoints of the network, so any attempt to build features in the network to support specific applications must be avoided or only viewed as a tradeoff. The reason was that not all applications need the same features and network functions to support them. Thus building such functions in the network core is rarely necessary. So, systems designers should avoid building any more than the essential and commonly shared functions into the network.

What were the designers’ original goals that led to the e2e principle?  Moving functions and services closer to the applications that use them increases the flexibility and the autonomy of the application designer to offer these services to the needs of the specific application. Thus, the higher-level protocol layers are more specific to an application. Whereas the lower-level protocol layers are free to organize the lower-level network resources to achieve application design goals more efficiently and independently of the specific application.
## What are the examples of a violation of e2e principle?  

Examples include **firewalls** and **traffic filters**. 

Firewalls usually operate at the periphery of a network, and they monitor the network traffic going through. They will either allow traffic to go through or drop traffic flagged as malicious. Firewalls violate the e2e principle since they are intermediate devices operated between two end hosts and can drop the end hosts' communication.

Another example of an e2e violation is the **Network Address Translation (NAT)** boxes. NAT boxes help us as a band-aid measure to deal with the shortage of internet addresses. Let's see in more detail how a NAT-enabled home router operates. Assume we have a home network with multiple devices we want to connect to the Internet. An internet service provider typically assigns a single public IP address (`120.70.39.4`) to the home router and specifically to the interface that is facing the public Internet, as shown in the figure below.

![[End to End Violation - NAT box diagram.png]]

The other interface of the NAT-enabled router facing the home network gets an IP address that belongs to the same private subnet. This subnet must belong to the address spaces that are reserved as private, e.g., 10.0.0/8 or 192.168.0.0/24. This means that the IP addresses that belong to this private subnet have meaning *only to devices within that subnet*. So we can have hundreds of thousands of private networks with the same address range (e.g., 10.0.0.0/8). But, these private networks are always behind a NAT, which takes care of the communication between the hosts on the private network and the hosts on the public Internet.

The home router plays the role of a translator maintaining a NAT translation table, and it rewrites the source and destination IP addresses and ports. The translation table provides a mapping between the public-facing IP address/ports and the IP addresses/ports that belong to hosts inside the private network. For example, let's assume that a host 10.0.0.4 inside the private network uses port 3345 to send traffic to a host in the public Internet with IP address 120.70.39.40 and port 5001 (see diagram below). 


![[Network Address Translation.png]]

The NAT table says that packets with the source IP address of 10.0.0.4 and source port 3345 will be rewritten with a source address 120.70.39.40 and a source port of 5001 (or any source port number not currently used in the NAT translation table). Similarly, packets with a destination IP address of 120.70.39.40 and destination port of 5001 will be rewritten to destination IP address 10.0.0.4 and destination port 3345.

#### **Why do NAT boxes violate the e2e principle?**

The hosts behind NAT boxes are not globally addressable or routable. As a result, it is not possible for other hosts on the public Internet to initiate connections to these devices. So, if we have a host behind a NAT and a host on the public Internet, they cannot communicate by default without the intervention of a NAT box. 

Some workarounds allow hosts to initiate connections to hosts that exist behind NATs. For example, Session Traversal Utilities for NAT, or STUN, is a tool that enables hosts to discover NATs and the public IP address and port number that the NAT has allocated for the applications for which the host wants to communicate. Also, UDP hole punching establishes bidirectional UDP connections between hosts behind NATs.