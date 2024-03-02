#incubator 
upstream: [[Software Development]]

---

**links**: 

---

Brain Dump: 

--- 

## Introduction

Researchers have suggested a model called the Evolutionary Architecture model, or **EvoArch**, that can help to study layered architectures and their evolution in a quantitative manner. Through this model, researchers were able to explain how the hierarchical structure of the layer architecture eventually led to the hourglass shape.

>the interpretation of the EvoArch model in the context of protocol evolution and survival essentially suggests that a protocol's longevity and dominance are influenced by its ability to maximize its products (applications and services that use the protocol) and its substrates (the underlying layers and technologies that support the protocol).

## Definition 

Evolutionary Architecture refers to an approach in software development that prioritizes flexibility, allowing systems to evolve and adapt as requirements change. This methodology is designed to accommodate the inevitable changes that occur in the technology landscape and business requirements.

## Key Principles

### 1. Incremental Change
- **Description**: Evolutionary Architecture supports making small, incremental changes rather than large-scale overhauls.
- **Benefits**: This approach reduces risk and allows for more manageable and predictable progress.

### 2. Guided by Fitness Functions
- **Fitness Functions**: These are specific objectives or criteria that guide the evolution of the architecture. They can be quantitative or qualitative measures of system performance, maintainability, scalability, etc.
- **Application**: Fitness functions help in objectively assessing whether changes are moving the system in the desired direction.

### 3. Modular and Component-Based Design
- **Modularity**: The system is designed in a way that allows individual components or modules to be updated independently.
- **Advantages**: Enhances flexibility and makes it easier to implement changes without impacting the entire system.

### 4. Embracing Change
- **Expecting Change**: Evolutionary Architecture operates under the assumption that change is inevitable and designs systems to be as adaptable as possible.
- **Adaptation Strategies**: This may involve using patterns like microservices, which allow for parts of the system to be updated without affecting others.

## The "Hour Glass": 

The Evolutionary Architecture model (EvoArch) provides a compelling explanation for the emergence of the "hourglass shape" in layered architectures, such as the Internet's protocol stack. This hourglass shape is characterized by a wide variety of protocols at the top and bottom layers, but a narrow convergence around a few key protocols in the middle layers. Let's delve into how the EvoArch model explains this phenomenon:

![[Internet Hourglass Shape.png]]

1. **Top Layer Diversification**: In the upper layers of a layered architecture like the Internet, many new applications and application-specific protocols are continuously created, and few of them become obsolete or extinct. This constant innovation and limited competition in the top layers lead to a broad and diverse range of protocols, widening the top of the hourglass.

2. **Middle Layer Convergence (Evolutionary Kernels)**: EvoArch predicts the emergence of a few powerful and longstanding protocols in the middle layers, referred to as evolutionary kernels. These include protocols like IPv4 in the network layer and TCP and UDP in the transport layer. These kernel protocols provide a stable framework for a vast array of physical and data-link layer protocols at the bottom, as well as new applications and services at the top layers. They become difficult to replace or modify significantly due to their entrenched position and widespread use.
	
	- In the middle layers, especially in layers like the Network and Transport layers of the OSI model, there is a critical need for standard protocols that ensure interoperability between different systems and networks.
	- Protocols like TCP/IP in these layers become dominant because they provide a common language that allows diverse systems to communicate effectively. This standardization is essential for the global interconnectedness of networks.
	- Abstraction in system design simplifies complexity by hiding lower-level details. For a network to function efficiently, it's impractical for every application or service at the higher layers to deal with the intricacies of every possible lower-layer technology.
	- Dominant protocols at the middle layers abstract these details, presenting a simpler, unified interface to the higher layers. This reduces complexity and fosters the development of applications and services that can operate over a wide range of underlying physical and data-link technologies.

3. **Bottom Layer Variety**: At the bottom layer, each protocol serves as a general building block, and because they are used abundantly, none of them dominate. This leads to a variety of protocols coexisting at the bottom layer, widening the bottom of the hourglass. For example, in the physical layer, multiple technologies coexist (optical fibers, copper cables, wireless signals), each serving different networking needs. In the middle layers, a few protocols become highly dominant because they need to provide universal interoperability and thus have less variety. This contrast in protocol diversity between the bottom and middle layers creates the hourglass shape.

4. **Quality Factor in EvoArch**: When a quality factor, capturing protocol performance, deployment extent, reliability, or security, is included in the EvoArch model, the network continues to exhibit an hourglass shape. However, this factor influences the competition primarily in the bottom layers, ensuring only high-quality protocols survive there. Interestingly, the model shows that the kernel protocols at the waist of the hourglass are not necessarily the highest-quality protocols but rather those created early with the right set of connections.

5. **Broader Implications**: Researchers are also applying the EvoArch model to explore the emergence of hourglass architectures in other areas like metabolic and gene regulatory networks, the organization of the innate immune system, and gene expression during development. This suggests a broader applicability of the hourglass model beyond just technological systems like the Internet.

In summary, the EvoArch model illustrates how layered architectures, through a process akin to natural evolution, tend to develop a wide range of protocols at the top and bottom layers, but converge around a few key protocols in the middle layers, creating an hourglass shape. This structure is not necessarily a reflection of the quality of the protocols but is influenced by their early establishment and interconnectedness within the network

## Best Practices

1. **Continuous Integration and Delivery (CI/CD)**: Regularly and automatically testing and deploying changes to ensure that the system is always in a deployable state.
2. **Refactoring**: Continuously improving the internal structure of the system without changing its external behavior.
3. **Automated Testing**: Ensuring a comprehensive suite of tests that can quickly identify issues introduced by changes.

## Challenges and Considerations

- **Balancing Flexibility and Stability**: While adaptability is key, it's important to maintain a level of stability in the system.
- **Long-term Vision**: Understanding the broader business goals and technological trends to guide the evolution effectively.
- **Technical Debt Management**: Being vigilant about technical debt, which can accumulate and hinder the system's ability to evolve.

---

## **EvoArch iterations**

EvoArch is a discrete-time model that is executed over rounds. At each round, we perform the following steps:

1. We introduce new nodes, and we place them randomly within the layers.
2. We examine all layers, from the top to the bottom, and we perform the following tasks:
    1. We connect the new nodes that we may have just introduced to that layer by choosing substrates based on the generality probabilities of the layer below s(l−1) and by choosing products for them based on the generality probability of the current layer s(l).
    2. We update the value of each node at each layer l, given that we may have new nodes added to the same layer l.
    3. We examine all nodes in order of decreasing value in that layer and remove the nodes that should die.
3. Finally, we stop the execution of the model when the network reaches a given number of nodes.

#### **Implications for the Internet Architecture and future Internet architecture**

With the help of the EvoArch model, how can we explain the survival of the TCP/IP stack, given that it appeared around the 70s or 80s when the telephone network was very powerful? The EvoArch model suggests that the TCP/IP stack was not trying to compete with the telephone network services. The TCP/IP was mostly used for applications such as FTP, E-mail, and Telnet, so it managed to grow and increase its value without competing or being threatened by the telephone network at the time that it first appeared. Later it gained even more traction, with numerous and powerful applications relying on it.   

IPv4, TCP, and UDP provide a stable framework through which there is an ever-expanding set of protocols at the lower layers (physical and data link layers), as well as new applications and services at the higher layers. But at the same time, these same protocols have been difficult to replace or even modify significantly. EvoArch provides an explanation for this. A large birth rate at the layer above the waist can cause death for the protocols at the waist if these are not chosen as substrates by the new nodes at the higher layers. The waist of the Internet architecture is narrow, but the next higher layer (the transport layer) is also very narrow and stable. So, the transport layer acts as an “evolutionary shield” for IPv4 because any new protocols that might appear at the transport layer are unlikely to survive the competition with TCP and UDP, which already have multiple products. In other words, the stability of the two transport protocols adds to the stability of IPv4 by eliminating any potential new transport protocols that could select a new network layer protocol instead of IPv4.

Finally, in terms of future and entirely new Internet architectures, the EvoArch model predicts that even if these brand-new architectures do not have the shape of an hourglass initially, they will probably do so as they evolve, which will lead to new ossified protocols. The model suggests that one way to proactively avoid these ossification effects that we now experience with TCP/IP is for a network architect to design the functionality of each layer so that the waist is wider, consisting of several protocols that offer largely non-overlapping but general services, so that they do not compete with each other.