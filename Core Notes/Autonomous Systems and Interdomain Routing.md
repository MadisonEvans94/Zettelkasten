#evergreen1 
upstream:

---

**links**: [AS](https://www.youtube.com/watch?v=-pYf2NqrL_I&ab_channel=SunnyClassroom), [BGP](https://www.youtube.com/watch?v=O6tCoD5c_U0&ab_channel=Computerphile)

---
## What Is An Autonomous System?
An **Autonomous System (AS)** is an independent network or system of networks that are controlled by a single entity  like ISPs/governments/companies. An Autonomous System is a network of routers that all use the same routing logic and policy. Each AS has its unique ID: **AS Number (ASN)**. The ASN is a globally unique 16-bit number (i/e: 1-65.534). The ASN is assigned by the [Internet Assigned Numbers Authority (IANA)](https://www.iana.org/). 

![[AS Graphic.png]]
## ISPs

The Internet as we know it today is an intricate ecosystem comprising a vast network of interconnected networks. This complex structure is supported by several key components, including **Internet Service Providers (ISPs)**, **Internet Exchange Points (IXPs)**, and **Content Delivery Networks (CDNs)**. Each plays a critical role in delivering digital content across the globe, ensuring that users can access information and services seamlessly.

### Internet Service Providers (ISPs)

ISPs are organizations that provide a variety of services necessary for accessing and utilizing the Internet. They serve as the gateway for individuals and businesses to connect to the global network, offering services such as web access, domain name registration, hosting, and email services. ISPs can be categorized into different tiers, reflecting their place and role within the global network hierarchy:

- **Tier 1 ISPs:** These are the top-level ISPs that form the backbone of the Internet. They have a global reach and can exchange traffic without paying any fees through peering agreements. Tier 1 ISPs interconnect with each other through high-capacity networks, ensuring that data can traverse the globe without restrictions.

- **Tier 2 ISPs:** These ISPs have regional or national coverage and typically engage in peering arrangements with other Tier 2 ISPs, as well as purchasing transit or upstream bandwidth from Tier 1 providers. They balance between peering and paying for access to parts of the Internet not within their network.

- **Tier 3 ISPs:** Often referred to as local ISPs, Tier 3 providers usually offer Internet access to end-users and businesses. They generally purchase Internet transit from Tier 2 or Tier 1 ISPs to connect to the broader Internet. This tier represents the last mile of the Internet, directly interacting with consumers.

### Protocols for Routing Traffic

The operation of the Internet relies heavily on protocols designed for routing traffic efficiently across different Autonomous Systems (ASes) and within them. These protocols are categorized into **Exterior Gateway Protocols (EGPs)** and **Interior Gateway Protocols (IGPs)**:
![[Interior Router vs Border Router.png]]

- **Border Gateway Protocol (BGP):** BGP is the de facto EGP that facilitates data routing between ASes, enabling different parts of the Internet to communicate. It is used by border routers of ASes to exchange routing information, helping to determine the best paths for data transmission across the Internet's vast landscape.

- **Interior Gateway Protocols (IGPs):** Within an AS, IGPs take charge, focusing on optimizing path metrics to ensure efficient data flow internally. Common IGPs include:
  - **Open Shortest Path First (OSPF):** A widely used IGP that employs a link-state routing algorithm to find the best path between the source and destination.
  - **Intermediate System to Intermediate System (IS-IS):** Similar to OSPF, IS-IS also uses a link-state routing protocol but is known for its scalability and efficiency in large networks.
  - **Routing Information Protocol (RIP):** One of the oldest routing protocols, using a distance-vector routing algorithm to determine the best path based on hop count.
  - **Enhanced Interior Gateway Routing Protocol (E-IGRP):** A Cisco proprietary protocol that improves on IGP routing efficiency by using a combination of distance-vector and link-state features.

This section focuses on BGP due to its critical role in maintaining the Internet's global connectivity. Understanding BGP and its interaction with IGPs is essential for comprehending how data is routed through the complex network of networks that make up the Internet.

## AS Business Relationships 

Autonomous Systems (ASes) interact with each other through various types of business relationships, primarily to facilitate the exchange of traffic in the most efficient and cost-effective manner possible. These relationships are fundamental to the operation of the Internet, dictating the flow of data and influencing the architecture of the global network.

1. **Provider-Customer Relationship (Transit):** This relationship forms the backbone of Internet connectivity, where the customer pays the provider for access to the entire Internet. The provider agrees to forward the customer's traffic to all possible destinations in the provider's routing table and accept traffic from anywhere destined for the customer. This model is prevalent across all tiers of ISPs, enabling smaller networks to connect to the larger Internet ecosystem. The financial settlement is typically based on the volume of traffic sent or received, bandwidth capacity, or a flat rate, providing a predictable cost structure for the customer.

2. **Peering Relationship:** Peering is a mutually beneficial arrangement where two ASes agree to exchange traffic between their respective customers without any financial settlement. This direct exchange can significantly reduce latency, improve bandwidth efficiency, and decrease dependency on transit providers. Peering is most common among ISPs of similar size and traffic volume to ensure a balance of benefits. While traditionally associated with Tier-1 ISPs, peering has become increasingly popular among smaller networks, facilitated by Internet Exchange Points (IXPs) where multiple ASes can interconnect directly.

**Financial Models in AS Relationships:**

- **Fixed Pricing:** A straightforward model where the customer pays a set fee for a specified amount of bandwidth, regardless of actual usage. This model is suitable for customers with predictable traffic patterns.

- **Usage-Based Pricing:** More dynamic than fixed pricing, this model charges based on the volume of traffic sent through the provider's network. The 95th percentile billing method is commonly used, offering a balance between flexibility and predictability. It allows for occasional traffic spikes without a corresponding spike in cost.

**Influences on Routing Policies:**

Economic considerations heavily influence routing policies within ASes. Providers may implement policies that encourage or incentivize increased traffic flow from their customers to maximize revenue. Similarly, peering arrangements are designed to balance traffic levels to avoid potential disputes over asymmetrical traffic flows.

## BGP Routing Protocols

The Border Gateway Protocol (BGP) is essential for routing traffic across the Internet, enabling data to travel across various autonomous systems. BGP is divided into two main types: External BGP (eBGP) and Internal BGP (iBGP).

### Importing Routes

When an AS receives route announcements via BGP, it must decide which routes to import into its routing table. This decision is based on policies that may consider factors such as the route's origin, path attributes, and any agreements with the neighboring AS. These policies help ensure that the AS selects the most efficient or cost-effective paths.

### Exporting Routes

Similarly, an AS must decide which routes to advertise to its neighbors. This includes determining which routes are shared with customers, providers, and peers. Export policies can be influenced by business relationships, with the aim of optimizing traffic flow, minimizing transit costs, or maintaining balanced peering relationships.

**eBGP and iBGP:**

- **eBGP:** Used for exchanging routing information between different ASes. It is the protocol that enables the Internet to function as a network of networks, determining the best paths between ASes.

- **iBGP:** Operates within a single AS, distributing routes learned from eBGP to all routers within the AS. Unlike eBGP, iBGP does not require routers to be directly connected, as long as they can communicate through the AS's internal network.

**BGP Messages:**

- **Updates:** Announce new routes or withdraw previously announced routes. These messages are crucial for maintaining an up-to-date view of the global routing table.

- **Keepalive:** Sent periodically to ensure that the connection between BGP peers remains active. If a router stops receiving these messages, it will assume the connection has been lost and will remove the peer's routes from its routing table.

Enhancing the understanding of AS business relationships and BGP routing protocols provides a clearer picture of the complex interactions that underpin the global Internet. This expanded insight into the financial models, peering dynamics, and technical mechanisms of routing illustrates the multifaceted nature of Internet connectivity.

## IXPs

Internet Exchange Points (IXPs) are key components of the Internet infrastructure, enabling networks to interconnect directly through peering, which reduces the dependency on transit services and often improves end-to-end connection speed and reliability. At the core of IXPs are two types of servers that play pivotal roles in network communication and data routing: root servers and route servers.

### Root Servers

Root servers are a crucial element of the global Domain Name System (DNS), responsible for translating human-friendly domain names into IP addresses that computers use to identify each other on the Internet. Although not physically located at IXPs, their function is foundational to all Internet services, including those facilitated by IXPs.

- **Functionality:** When an Internet user wants to visit a website, their computer first queries a DNS resolver. If the resolver does not have the IP address cached, it queries one of the root servers. The root server responds with a referral to the Top-Level Domain (TLD) servers (e.g., .com, .net, .org) responsible for the second-level domain of the requested URL. The process continues down the DNS hierarchy until the IP address is found, allowing the user's computer to establish a connection to the desired web server.

- **Resilience and Distribution:** There are 13 root server addresses, operated by 12 different organizations worldwide. These servers are replicated using anycast routing, allowing multiple physical servers across the globe to share the same IP address, thus increasing the resilience and reliability of DNS queries by routing requests to the nearest or most available server instance.

### Route Servers (RS)

Route servers at IXPs facilitate multilateral peering among multiple networks without the need for each network to establish bilateral peering arrangements with every other network. This simplifies the peering process and enhances the efficiency of data exchange at IXPs.

- **Multilateral Peering Sessions:** A route server allows a participating network to see routes from all other networks peering with the route server. When a network sends its routing information to the route server, the server redistributes this information to all other peering networks. This enables each network to understand the paths available through the IXP for reaching different destinations on the Internet.

- **Policy Implementation and Control:** Route servers are configured with policies that reflect the peering agreements and preferences of the participating networks. These policies determine how routes are advertised and accepted between the peering parties. By using a route server, networks can implement sophisticated policies that control traffic flow, ensuring that data takes optimal paths according to the network's strategic and economic interests.

- **Advantages of Using Route Servers:** By centralizing route exchange, route servers reduce the complexity and operational overhead associated with managing multiple peering sessions. This is particularly beneficial for smaller networks seeking to maximize their connectivity options. Additionally, route servers enhance the scalability of peering at IXPs by making it easier to add or modify peering relationships without direct configuration changes on each router.

IXPs, through the facilitation of root servers and route servers, play a pivotal role in the structure and performance of the Internet. Root servers maintain the hierarchy and efficiency of the DNS, ensuring users can access websites using memorable domain names instead of numeric IP addresses. Route servers streamline and optimize the process of network interconnection, allowing for a more connected, efficient, and resilient Internet.
