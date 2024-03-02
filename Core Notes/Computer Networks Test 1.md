#temp
upstream:

---

**links**: 

---
Exam 1 covers Lessons 1 through Lesson 6. It will be open during the end of Week 7 (Feb 26 - Mar 3 AOE).

The following questions and prompts are intended to check your understanding of the content. The exam may cover ANY content from the modules. The only exception is those pages marked “optional.”

You can find the study guide for all lessons on Canvas. This post serves as a location where you can receive clarification and have discussions concerning the exam.

The course staff does not release official answers to these questions, but students may collaborate as long as any collaborative documents are locked during the exam quiet period. Happy studying!

## Lesson 1: Introduction, History, and Internet Architecture

- What are the advantages and disadvantages of a layered architecture?
    
- What are the differences and similarities between the OSI model and the five-layered Internet model?
    
- What are sockets?
    
- Describe each layer of the OSI model.
    
- Provide examples of popular protocols at each layer of the five-layered Internet model.
    
- What is encapsulation, and how is it used in a layered model?
    
- What is the end-to-end (e2e) principle?
    
- What are the examples of a violation of e2e principle?
    
- What is the EvoArch model?
    
- Explain a round in the EvoArch model.
    
- What are the ramifications of the hourglass shape of the internet?
    
- Repeaters, hubs, bridges, and routers operate on which layers?
    
- What is a bridge, and how does it “learn”?
    
- What is a distributed algorithm?
    
- Explain the Spanning Tree Algorithm.
    
- What is the purpose of the Spanning Tree Algorithm?
    

## Lesson 2: Transport and Application Layers

- What does the transport layer provide?
    
- What is a packet for the transport layer called?
    
- What are the two main protocols within the transport layer?
    
- What is multiplexing, and why is it necessary?
    
- Describe the two types of multiplexing/demultiplexing.
    
- What are the differences between UDP and TCP?
    
- When would an application layer protocol choose UDP over TCP?
    
- Explain the TCP Three-way Handshake.
    
- Explain the TCP connection tear down.
    
- What is Automatic Repeat Request or ARQ?
    
- What is Stop and Wait ARQ?
    
- What is Go-back-N?
    
- What is selective ACKing?
    
- What is fast retransmit?
    
- What is transmission control, and why do we need to control it?
    
- What is flow control, and why do we need to control it?
    
- What is congestion control?
    
- What are the goals of congestion control?
    
- What is network-assisted congestion control?
    
- What is end-to-end congestion control?
    
- How does a host infer congestion?
    
- How does a TCP sender limit the sending rate?
    
- Explain Additive Increase/Multiplicative Decrease (AIMD) in the context of TCP.
    
- What is a slow start in TCP?
    
- Is TCP fair in the case where connections have the same RTT? Explain.
    
- Is TCP fair in the case where two connections have different RTTs? Explain.
    
- Explain how TCP CUBIC works.
    
- Explain TCP throughput calculation.
    

## Lesson 3: Intra-domain Routing

- [x] What is the difference between forwarding and routing?
Forwarding is the fast, hardware level action that involves a router receiving an incoming packet and sending it off to the next router; it's sending a packet from and incoming link to an outgoing link. Routing on the other hand is the more complex, high level process of determining the sequence of routers to hit in order to get a packet from point A to point B. It's a more complex process that often involves software level algorithms. Forwarding occurs at the **data plane** level while routing occurs at the **control plane** level 
- [x] What is the main idea behind a link-state routing algorithm?
- [x] What is the main idea behind the distance vector routing algorithm?
- [x] Walk through an example of the distance vector algorithm.
The main idea behind the link-state algorithm is that each node in the network has complete, up-to-date information about the entire network's topology including the cost to each node. It uses this information to find the shortest path to other nodes via algorithms such as **[[Dijkstra's Algorithm]]**. The process involves several key steps: 

1. **Dissemination of Information**: each node forwards information about its immediate neighbors to all other nodes until all nodes have received the information 
2. **Computation of Shortest Path**: Each node uses dijkstra's algorithm to calculate the shortest path to every other node. The algorithm starts with the node itself and then builds the shortest path tree until all nodes are included 
3. **Update and Repeat**: Nodes periodically broadcast their nearest neighbor link costs and update accordingly, making the process dynamic 

The fundamental difference between Link-State and Distance Vector Routing Algorithms lies in how they operate and their scalability and performance:

- **Link-State** algorithms require each node to have *complete knowledge* of the network's topology and then independently calculate the shortest paths. This results in more rapid convergence and avoids routing loops but requires *more memory and processing power* due to the need to store the entire topology and continually run Dijkstra's algorithm.

- **Distance Vector** algorithms, on the other hand, involve each node sharing information only with its neighbors. Each node updates its own routing table based on information from its neighbors and then shares its updated table, in a process that repeats until convergence. Distance vector algorithms are simpler and require less memory but can be slower to converge and are more susceptible to routing loops and the count-to-infinity problem.

Imagine a network as a city where every router is a post office. The goal is for every post office (router) to send mail (data packets) to the correct destination in the most efficient way possible.

**Link-State Algorithm:** In this scenario, each post office has a map of the entire city, including all the roads (links) and the condition of each road (bandwidth, congestion, etc.). Every time there is a change in the road conditions, an update is sent to every post office so they can update their maps. When a post office needs to send mail, it looks at its map, calculates the fastest route to the destination post office, and sends the mail along that route. This is like the Link-State Routing Algorithm, where routers have complete knowledge of the network topology and state, and independently calculate the shortest path to every other router.

**Distance Vector Algorithm:** In contrast, each post office doesn't have a complete map of the city. Instead, it only knows the distance and direction to its neighboring post offices and the best route to get to every other post office according to the information received from its neighbors. Periodically, each post office tells its neighbors how far away every other post office is from its perspective. Based on this information, a post office updates its own best routes to every destination, sometimes learning better routes through its neighbors. This is akin to the Distance Vector Routing Algorithm, where routers only know the distance (cost) to reach a destination via their immediate neighbors and update their routing tables based on information received from those neighbors.

Distance Vector algorithms are generally considered better suited for smaller or less complex networks rather than really large ones. This is because, although Distance Vector algorithms have lower overhead in terms of the amount of information that needs to be exchanged initially (since nodes only share information with their immediate neighbors), they suffer from slower convergence times, especially in larger networks or in the case of network topology changes. This slow convergence can lead to temporary routing loops and potentially inconsistent routing tables across the network until all nodes have updated their information.

On the other hand, Link-State algorithms, despite requiring a higher initial overhead due to the exchange of information about the entire network topology among all nodes, are more efficient for larger networks. This is because they tend to converge faster and are more reliable in maintaining consistent and loop-free paths, thanks to each node having a complete view of the network and using a deterministic algorithm (like Dijkstra's) to compute the shortest path.

So, while Distance Vector algorithms are simpler and require less immediate information exchange, making them appealing for smaller or simpler networks, Link-State algorithms are generally preferred for larger, more complex networks due to their faster convergence and greater reliability.

- [x] What is an example of a link-state routing algorithm? Walk through an example
[[OSPF]]
- [x] What is the computational complexity of the link-state routing algorithm?
The computational complexity of the link state routing algorithm is in the order of $O(n^2)$. In the worst-case scenario, the number of computations needed to find the least-cost paths from the source to all destinations in the network follows a sequence where the first iteration requires searching through all nodes to find the node with the minimum path cost. In subsequent iterations, the number of nodes to search through decreases by one each time (from $(n)$ to $(n-1)$, and so on), resulting in a total of $(n(n+1)/2)$ nodes searched through by the end of the algorithm
- [x] When does the count-to-infinity problem occur in the distance vector algorithm?
- [x] How does poison reverse solve the count-to-infinity problem?
- [x] What is the Routing Information Protocol (RIP)?

**The Routing Information Protocol (RIP)** is a distance vector routing protocol that uses hop count as its metric for route selection, assuming the cost of each link as 1. It is designed for use within an **Autonomous System (AS)** and is characterized by its simplicity and ease of configuration. RIP operates by exchanging routing updates between neighbors periodically through RIP response messages, also known as RIP advertisements. These advertisements contain information about the router's distances to destination subnets.

Routers maintain a routing table that includes their own distance vector and the router's forwarding table. The routing table lists each subnet within the AS, the identification of the next router along the shortest path to the destination, and the number of hops to reach the destination along this path. When a router receives an advertisement from a neighbor, it merges this new information with its existing routing table, potentially updating routes to reflect shorter paths discovered through the received advertisements.

RIP faces challenges such as updating routes, reducing convergence time, and avoiding routing loops, including the count-to-infinity problem. To mitigate issues related to routers becoming unreachable, RIP considers a neighbor as no longer reachable if it does not hear from it at least once every 180 seconds, prompting an update in the local routing table and propagation of these changes. RIP version 2 introduced improvements such as the ability to aggregate subnet entries using route aggregation techniques, enhancing its scalability and efficiency. RIP messages are exchanged over UDP, using port number 520, and the protocol is implemented as an application-level process

- [x] What is the Open Shortest Path First (OSPF) protocol?
- [x] How does a router process advertisements?
- [x] What is hot potato routing?
**Processing OSPF Messages in a Router:**
1. **Initial Receipt of LS Update Packets:** These packets contain Link State Advertisements (LSAs) from neighboring routers and are processed by the route processor of the router. This forms a consistent view of the topology, stored in the link-state database.
2. **Calculation of Shortest Path:** The router uses the information from the link-state database to calculate the shortest path using the Shortest Path First (SPF) algorithm. The results are then used to update the Forwarding Information Base (FIB).
3. **Forwarding Data Packets:** The FIB is utilized when a data packet arrives at an interface card of the router. It determines the next hop for the packet and forwards it to the outgoing interface card【47†source】.

**Hot Potato Routing** is a routing technique used in large networks where routers rely on both interdomain and intradomain routing protocols. This technique involves choosing the closest egress point based on intradomain path costs (Interior Gateway Protocol/IGP costs) when multiple egress points are available and equally good in terms of external path quality to the destination. Hot potato routing simplifies computations for routers, ensures path consistency, and reduces network resource consumption by routing the traffic out of the network as quickly as possible【48†source】.
## Lesson 4: AS Relationships and Inter-domain Routing

- [x] Describe the relationships between ISPs, IXPs, and CDNs.
- **ISPs (Internet Service Providers):** ISPs are companies that provide individuals and other companies access to the Internet and other related services. They form the backbone of the internet by connecting end-users to the global network. ISPs can range from local providers serving a small community to large multinational companies that offer extensive global coverage.

- **IXPs (Internet Exchange Points):** IXPs are key components of the Internet infrastructure where multiple ISPs and other network providers connect to exchange traffic between their networks. IXPs reduce the portion of an ISP's traffic that must be delivered via their upstream transit providers, thereby reducing the average per-bit delivery cost of their service. They also improve the speed and reliability of the internet by keeping local internet traffic within local infrastructure and reducing the distances it must travel.

- **CDNs (Content Delivery Networks):** CDNs are networks of servers strategically located across different geographical locations, designed to deliver web content and services to users more efficiently and quickly. CDNs store cached content (like web videos, images, and web pages) on edge servers close to the end-users to minimize latency.


The relationship between these entities can be described as follows:

1. **ISPs and IXPs:** ISPs connect to IXPs to exchange traffic with other networks. This setup enables ISPs to route traffic more efficiently, improving internet speed and reducing bandwidth costs. By peering at IXPs, ISPs can directly exchange traffic without needing to send traffic through upstream providers, enhancing the overall efficiency of internet connectivity.

2. **ISPs and CDNs:** ISPs and CDNs work together to deliver content to end-users efficiently. CDNs place their content servers inside or close to ISPs' networks to reduce the distance data must travel to reach the end-user, which reduces latency and bandwidth costs. This collaboration benefits both parties: ISPs can offer faster, more reliable access to popular content, and CDNs can ensure their content is delivered as efficiently as possible.

3. **IXPs and CDNs:** CDNs often connect to IXPs to distribute their content more broadly and efficiently. By peering at IXPs, CDNs can exchange traffic directly with multiple ISPs and other networks connected to the IXP, enhancing content delivery speed and reliability to end-users across different networks.

Together, ISPs, IXPs, and CDNs form a symbiotic relationship that enhances the performance, reliability, and efficiency of the internet, ensuring that users can access content quickly and reliably regardless of their location.

- [x] What is an AS? What kind of relationship does AS have with other parties?
An **Autonomous System (AS)** is a group of routers, including the links among them, that operates under the same administrative authority. ISPs and CDNs, for example, may operate as an AS. An AS can be a single entity or it may consist of multiple ASes. Within an AS, a unified set of policies is implemented, traffic engineering decisions are made, and strategies for interconnection are determined. These policies and strategies dictate how traffic enters and exits the network.

ASes use different protocols for routing traffic internally and externally:
- **Internal Gateway Protocols (IGPs)** are used within an AS to optimize path metrics. Examples of IGPs include Open Shortest Path First (OSPF), Intermediate System to Intermediate System (IS-IS), Routing Information Protocol (RIP), and Enhanced Interior Gateway Routing Protocol (E-IGRP).
- **Border Gateway Protocol (BGP)** is used by the border routers of ASes to exchange routing information with one another. BGP allows ASes to communicate their network reachability to other ASes, facilitating the routing of traffic across the internet's interconnected networks.

The relationship between ASes and other parties, such as ISPs, IXPs (Internet Exchange Points), CDNs, and other ASes, is crucial for the global routing infrastructure. ASes connect to one another directly or through IXPs, forming complex relationships based on various factors, including traffic volume, geographic location, and financial agreements. These relationships can be categorized into peering and transit agreements, where ASes exchange traffic either on a settlement-free basis (peering) or through a financial agreement where one AS pays another for the transit of traffic (transit). The structure and policies of these relationships significantly impact the efficiency, reliability, and cost of internet traffic routing【86†source】.
- [x] What is BGP? What are the basics of BGP? What were the original design goals of BGP? What was considered later?
[[Autonomous Systems and Interdomain Routing]]
- [ ] How does an AS determine what rules to import/export?
- [ ] What is the difference between iBGP and eBGP?
- [ ] What is the difference between iBGP and IGP-like protocols (RIP or OSPF)?
- [ ] How does a router use the BGP decision process to choose which routes to import?
- [ ] What are the 2 main challenges with BGP? Why?
- [ ] What is an IXP?
- [ ] What are four reasons for IXP's increased popularity?
- [ ] Which services do IXPs provide?
- [ ] How does a route server work?
## Lesson 5: Router Design and Algorithms (Part 1)

- [ ] What are the basic components of a router?
- [ ] Explain the forwarding (or switching) function of a router.
- [ ] The switching fabric moves the packets from input to output ports. What are the functionalities performed by the input and output ports?
- [ ] What is the purpose of the router’s control plane?
- [ ] What tasks occur in a router?
- [ ] List and briefly describe each type of switching. Which, if any, can send multiple packets across the fabric in parallel?
- [ ] What are two fundamental problems involving routers, and what causes these problems?
- [ ] What are the bottlenecks that routers face, and why do they occur?
- [ ] Convert between different prefix notations (dot-decimal, slash, and masking).
- [ ] What is CIDR, and why was it introduced?
- [ ] Name 4 takeaway observations around network traffic characteristics. Explain their consequences.
- [ ] Why do we need multibit tries?
- [ ] What is prefix expansion, and why is it needed?
- [ ] Perform a prefix lookup given a list of pointers for unibit tries, fixed-length multibit ties, and variable-length multibit tries.
- [ ] Perform a prefix expansion. How many prefix lengths do old prefixes have? What about new prefixes?
- [ ] What are the benefits of variable-stride versus fixed-stride multibit tries?
## Lesson 6: Router Design and Algorithms (Part 2)

- [ ] Why is packet classification needed?
- [ ] What are three established variants of packet classification?
- [ ] What are the simple solutions to the packet classification problem?
- [ ] How does fast searching using set-pruning tries work?
- [ ] What’s the main problem with the set pruning tries?
- [ ] What is the difference between the pruning approach and the backtracking approach for packet classification with a trie?
- [ ] What’s the benefit of a grid of tries approach?
- [ ] Describe the “Take the Ticket” algorithm.
- [ ] What is the head-of-line problem?
- [ ] How is the head-of-line problem avoided using the knockout scheme?
- [ ] How is the head-of-line problem avoided using parallel iterative matching?
- [ ] Describe FIFO with tail drop.
- [ ] What are the reasons for making scheduling decisions more complex than FIFO?
- [ ] Describe Bit-by-bit Round Robin scheduling.
- [ ] Bit-by-bit Round Robin provides fairness; what’s the problem with this method?
- [ ] Describe Deficit Round Robin (DRR).
- [ ] What is a token bucket shaping?
- [ ] In traffic scheduling, what is the difference between policing and shaping?
- [ ] How is a leaky bucket used for traffic policing and shaping?




