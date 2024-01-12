#seed 
upstream: [[Network Engineering]]

---

**links**: 

---

brain dump: 

--- 




# CS 6250 Computer Networks Study Questions  
The following questions and prompts are intended to check your understanding of the content. The  
exam may cover ANY content from the modules. The only exception being those pages marked  
“optional”.  

## Lesson 1: Introduction, History, and Internet Architecture  
- [ ] What are the advantages and disadvantages of a layered architecture?  
- [ ] What are the differences and similarities between the OSI model and the five-layered Internet  
model?  
- [ ] What are sockets?  
- [ ] Describe each layer of the OSI model.  
- [ ] Provide examples of popular protocols at each layer of the five-layered Internet model.  
- [ ] What is encapsulation, and how is it used in a layered model?  
- [x]  [[What is the end-to-end (e2e) principle?]]  
- [x] What are the examples of a violation of e2e principle?  
- [x]  [[What is the EvoArch model?]]
- [ ] Explain a round in the EvoArch model.  
- [ ] What are the ramifications of the hourglass shape of the internet?  
- [ ] Repeaters, hubs, bridges, and routers operate on which layers?  
- [ ] What is a bridge, and how does it “learn”?  
- [ ] What is a distributed algorithm?  
- [ ]  Explain the Spanning Tree Algorithm.  
- [ ]  What is the purpose of the Spanning Tree Algorithm?

## Lesson 2: Transport and Application Layers  
- [ ] What does the transport layer provide?  
- [ ]  What is a packet for the transport layer called?  
- [ ] What are the two main protocols within the transport layer?  
- [ ] What is multiplexing, and why is it necessary?  
- [ ] Describe the two types of multiplexing/demultiplexing.  
• What are the differences between UDP and TCP?  
• When would an application layer protocol choose UDP over TCP?  
• Explain the TCP Three-way Handshake.  
• Explain the TCP connection tear down.  
• What is Automatic Repeat Request or ARQ?  
• What is Stop and Wait ARQ?  
• What is Go-back-N?  
• What is selective ACKing?  
• What is fast retransmit?  
• What is transmission control, and why do we need to control it?  
• What is flow control, and why do we need to control it?  
• What is congestion control?  
• What are the goals of congestion control?  
• What is network-assisted congestion control?  
• What is end-to-end congestion control?  
• How does a host infer congestion?  
• How does a TCP sender limit the sending rate?  
• Explain Additive Increase/Multiplicative Decrease (AIMD) in the context of TCP.  
• What is a slow start in TCP?  
• Is TCP fair in the case where connections have the same RTT? Explain.  
• Is TCP fair in the case where two connections have different RTTs? Explain.  
• Explain how TCP CUBIC works.  
• Explain TCP throughput calculation.

## Lesson 3: Intradomain Routing  
• What is the difference between forwarding and routing?  
• What is the main idea behind a link-state routing algorithm?  
• What is an example of a link-state routing algorithm?  
• Walk through an example of the link-state routing algorithm.  
• What is the computational complexity of the link-state routing algorithm?  
• What is the main idea behind the distance vector routing algorithm?  
• Walk through an example of the distance vector algorithm.  
• When does the count-to-infinity problem occur in the distance vector algorithm?  
• How does poison reverse solve the count-to-infinity problem?  
• What is the Routing Information Protocol (RIP)?  
• What is the Open Shortest Path First (OSPF) protocol?  
• How does a router process advertisements?  
• What is hot potato routing?

## Lesson 4: AS Relationships and Interdomain Routing  
• Describe the relationships between ISPs, IXPs, and CDNs.  
• What is an AS?  
• What kind of relationship does AS have with other parties?  
• What is BGP?  
• How does an AS determine what rules to import/export?  
• What were the original design goals of BGP? What was considered later?  
• What are the basics of BGP?  
• What is the difference between iBGP and eBGP?  
• What is the difference between iBGP and IGP-like protocols (RIP or OSPF)?  
• How does a router use the BGP decision process to choose which routes to import?  
• What are the 2 main challenges with BGP? Why?  
• What is an IXP?  
• What are four reasons for IXP's increased popularity?  
• Which services do IXPs provide?  
• How does a route server work?

## Lesson 5: Router Design and Algorithms (Part 1)  
• What are the basic components of a router?  
• Explain the forwarding (or switching) function of a router.  
• The switching fabric moves the packets from input to output ports. What are the functionalities  
performed by the input and output ports?  
• What is the purpose of the router’s control plane?  
• What tasks occur in a router?  
• List and briefly describe each type of switching. Which, if any, can send multiple packets across  
the fabric in parallel?  
• What are two fundamental problems involving routers, and what causes these problems?  
• What are the bottlenecks that routers face, and why do they occur?  
• Convert between different prefix notations (dot-decimal, slash, and masking).  
• What is CIDR, and why was it introduced?  
• Name 4 takeaway observations around network traffic characteristics. Explain their  
consequences.  
• Why do we need multibit tries?  
• What is prefix expansion, and why is it needed?  
• Perform a prefix lookup given a list of pointers for unibit tries, fixed-length multibit ties, and  
variable-length multibit tries.  
• Perform a prefix expansion. How many prefix lengths do old prefixes have? What about new  
prefixes?  
• What are the benefits of variable-stride versus fixed-stride multibit tries?

## Lesson 6: Router Design and Algorithms (Part 2)  
• Why is packet classification needed?  
• What are three established variants of packet classification?  
• What are the simple solutions to the packet classification problem?  
• How does fast searching using set-pruning tries work?  
• What’s the main problem with the set pruning tries?  
• What is the difference between the pruning approach and the backtracking approach for packet  
classification with a trie?  
• What’s the benefit of a grid of tries approach?  
• Describe the “Take the Ticket” algorithm.  
• What is the head-of-line problem?  
• How is the head-of-line problem avoided using the knockout scheme?  
• How is the head-of-line problem avoided using parallel iterative matching?  
• Describe FIFO with tail drop.  
• What are the reasons for making scheduling decisions more complex than FIFO?  
• Describe Bit-by-bit Round Robin scheduling.  
• Bit-by-bit Round Robin provides fairness; what’s the problem with this method?  
• Describe Deficit Round Robin (DRR).  
• What is a token bucket shaping?  
• In traffic scheduling, what is the difference between policing and shaping?  
• How is a leaky bucket used for traffic policing and shaping?

## Lesson 7: SDN (Part 1)  
• What spurred the development of Software Defined Networking (SDN)?  
• What are the three phases in the history of SDN?  
• Summarize each phase in the history of SDN.  
• What is the function of the control and data planes?  
• Why separate the control from the data plane?  
• Why did the SDN lead to opportunities in various areas such as data centers, routing, enterprise  
networks, and research networks?  
• What is the relationship between forwarding and routing?  
• What is the difference between a traditional and SDN approach in terms of coupling of control  
and data plane?  
• What are the main components of an SDN network and their responsibilities?  
• What are the four defining features of an SDN architecture?  
• What are the three layers of SDN controllers?

## Lesson 8: SDN (Part 2)  
• Describe the three perspectives of the SDN landscape.  
• Describe the responsibility of each layer in the SDN layer perspective.  
• Describe a pipeline of flow tables in OpenFlow.  
• What’s the main purpose of southbound interfaces?  
• What are three information sources provided by the OpenFlow protocol?  
• What are the core functions of an SDN controller?  
• What are the differences between centralized and distributed architectures of SDN controllers?  
• When would a distributed controller be preferred to a centralized controller?  
• Describe the purpose of each component of ONOS (Open Networking Operating System) a  
distributed SDN control platform.  
• How does ONOS achieve fault tolerance?  
• What is P4?  
• What are the primary goals of P4?  
• What are the two main operations of P4 forwarding model?  
• What are the applications of SDN? Provide examples of each application.  
• Which BGP limitations can be addressed by using SDN?  
• What’s the purpose of SDX?  
• Describe the SDX architecture.  
• What are the applications of SDX in the domain of wide-area traffic delivery?

## Lesson 9: Internet Security  
• What are the properties of secure communication?  
• How does Round Robin DNS (RRDNS) work?  
• How does DNS-based content delivery work?  
• How do Fast-Flux Service Networks work?  
• What are the main data sources used by FIRE (FInding Rogue nEtworks) to identify hosts that  
likely belong to rogue networks?  
• The design of ASwatch is based on monitoring global BGP routing activity to learn the control  
plane behavior of a network. Describe 2 phases of this system.  
• What are 3 classes of features used to determine the likelihood of a security breach within an  
organization?  
• (BGP hijacking) What is the classification by affected prefix?  
• (BGP hijacking) What is the classification by AS-Path announcement?  
• (BGP hijacking) What is the classification by data plane traffic manipulation?  
• What are the causes or motivations behind BGP attacks?  
• Explain the scenario of prefix hijacking.  
• Explain the scenario of hijacking a path.  
• What are the key ideas behind ARTEMIS?  
• What are the two automated techniques used by ARTEMIS to protect against BGP hijacking?  
• What are two findings from ARTEMIS?  
• Explain the structure of a DDoS attack.  
• What is spoofing, and how is related to a DDoS attack?  
• Describe a Reflection and Amplification attack.  
• What are the defenses against DDoS attacks?  
• Explain provider-based blackholing.  
• Explain IXP blackholing.  
• What is one of the major drawbacks of BGP blackholing?

## Lesson 10: Internet Surveillance and Censorship  
• What is DNS censorship?  
• What are the properties of GFW (Great Firewall of China)?  
• How does DNS injection work?  
• What are the three steps involved in DNS injection?  
• List five DNS censorship techniques and briefly describe their working principles.  
• Which DNS censorship technique is susceptible to overblocking?  
• What are the strengths and weaknesses of the “packet dropping” DNS censorship technique?  
• What are the strengths and weaknesses of the “DNS poisoning” DNS censorship technique?  
• What are the strengths and weaknesses of the “content inspection” DNS censorship technique?  
• What are the strengths and weaknesses of the “blocking with resets” DNS censorship  
technique?  
• What are the strengths and weaknesses of the “immediate reset of connections” DNS  
censorship technique?  
• Our understanding of censorship around the world is relatively limited. Why is it the case? What  
are the challenges?  
• What are the limitations of main censorship detection systems?  
• What kind of disruptions does Augur focus on identifying?  
• How does Iris counter the issue of lack of diversity while studying DNS manipulation? What are  
the steps associated with the proposed process?  
• What are the steps involved in the global measurement process using DNS resolvers?  
• What metrics does Iris use to identify DNS manipulation once data annotation is complete?  
Describe the metrics. Under what condition do we declare the response as being manipulated?  
• How to identify DNS manipulation via machine learning with Iris?  
• How is it possible to achieve connectivity disruption using the routing disruption approach?  
• How is it possible to achieve connectivity disruption using the packet filtering approach?  
• Explain a scenario of connectivity disruption detection in the case when no filtering occurs.  
• Explain a scenario of connectivity disruption detection in the case of inbound blocking.  
• Explain a scenario of connectivity disruption detection in the case of outbound blocking.

## Lesson 11: Applications (Video)  
• Compare the bit rate for video, photos, and audio.  
• What are the characteristics of streaming stored video?  
• What are the characteristics of streaming live audio and video?  
• What are the characteristics of conversational voice and video over IP?  
• How does the encoding of analog audio work (in simple terms)?  
• What are the three major categories of VoIP encoding schemes?  
• What are the functions that signaling protocols are responsible for?  
• What are three QoS VoIP metrics?  
• What kind of delays are included in "end-to-end delay"?  
• How does "delay jitter" occur?  
• What are the mitigation techniques for delay jitter?  
• Compare the three major methods for dealing with packet loss in VoIP protocols.  
• How does FEC (Forward Error Correction) deal with the packet loss in VoIP? What are the  
tradeoffs of FEC?  
• How does interleaving deal with the packet loss in VoIP/streaming stored audio? What are the  
tradeoffs of interleaving?  
• How does the error concealment technique deal with the packet loss in VoIP?  
• What developments lead to the popularity of consuming media content over the Internet?  
• Provide a high-level overview of adaptive video streaming.  
• (Optional) What are two ways to achieve efficient video compression?  
• (Optional) What are the four steps of JPEG compression?  
• (Optional) Explain video compression and temporal redundancy using I-, B-, and P-frames.  
• (Optional) Why is video compression unable to use P-frames all the time?  
• (Optional) What is the difference between constant bitrate encoding and variable bitrate  
encoding (CBR vs VBR)?  
• Which protocol is preferred for video content delivery - UDP or TCP? Why?  
• What was the original vision of the application-level protocol for video content delivery, and  
why was HTTP chosen eventually?  
• Summarize how progressive download works.

• How to handle network and user device diversity?  
• How does the bitrate adaptation work in DASH?  
• What are the goals of bitrate adaptation?  
• What are the different signals that can serve as an input to a bitrate adaptation algorithm?  
• Explain buffer-filling rate and buffer-depletion rate calculation.  
• What steps does a simple rate-based adaptation algorithm perform?  
• Explain the problem of bandwidth over-estimation with rate-based adaptation.  
• Explain the problem of bandwidth under-estimation with rate-based adaptation.

## Lesson 12: Applications (CDNs and Overlay Networks)  
• What is the drawback to using the traditional approach of having a single, publicly accessible  
web server?  
• What is a CDN?  
• What are the six major challenges that Internet applications face?  
• What are the major shifts that have impacted the evolution of the Internet ecosystem?  
• Compare the “enter deep” and “bring home” approach to CDN server placement.  
• What is the role of DNS in the way CDN operates?  
• What are the two main steps in CDN server selection?  
• What is the simplest approach to selecting a cluster? What are the limitations of this approach?  
• What metrics could be considered when using measurements to select a cluster?  
• How are the metrics for cluster selection obtained?  
• Explain the distributed system that uses a 2-layered system. What are the challenges of this  
system?  
• What are the strategies for server selection? What are the limitations of these strategies?  
• What is consistent hashing? How does it work?  
• Why would a centralized design with a single DNS server not work?  
• What are the main steps that a host takes to use DNS?  
• What are the services offered by DNS, apart from hostname resolution?  
• What is the structure of the DNS hierarchy? Why does DNS use a hierarchical scheme?  
• What is the difference between iterative and recursive DNS queries?  
• What is DNS caching?  
• What is a DNS resource record?  
• What are the most common types of resource records?  
• Describe the DNS message format.  
• What is IP Anycast?  
• What is HTTP Redirection?