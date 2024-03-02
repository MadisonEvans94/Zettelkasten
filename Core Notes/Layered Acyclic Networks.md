#seed 
upstream:

---

**links**: 

---

Brain Dump: 

--- 

![[Protocol Stack.png]]In this section, we will talk about a model that attempts to answer our previous questions. Researchers have suggested a model called the Evolutionary Architecture model or EvoArch, which can help illustrate layered architectures and their evolution in a quantitative manner. The EvoArch model considers an abstract model of the internet’s protocol stack that has the following components:

- **Layers:** A protocol stack is modeled as a directed, acyclic network with L layers.

- **Nodes:** Each network protocol is represented as a node. The layer of a node _u_ is denoted by l(_u_).

- **Edges:** Dependencies between protocols are represented as directed edges.

- **Node incoming edges:** If a protocol _u_ at layer l uses the service provided by a protocol _w_ at the lower layer l−1, then this is represented by an “upwards” edge from _w_ to _u_.

- **Node substrates:** We refer to substrates of a node _u_, S(_u_), as the set of nodes that _u_ is using their services. Every node has at least one substrate, except the nodes at the bottom layer.

- **Node outgoing edges:** The outgoing edges from a node _u_ terminate at the products of _u_. The products of a node u are represented by P(u).

- **Layer generality:** Each layer is associated with a probability s(l), which we refer to as layer generality. A node _u_ at layer l+1 independently selects each node of layer l as the substrate with probability s(l). The layer generality decreases as we move to higher layers. Protocols at lower layers are more general in terms of their functions or provided services than protocols at higher layers. For example, in the case of the Internet protocol stack, layer 1 is highly general, and the protocols at this layer offer a general bit transfer service between two connected points, which most higher-layer protocols would use.

- **Node evolutionary value:** 
>Think "how dependent are other higher level protocols on me? If I were to change, how much of the entire stack would have to change? The more dependence, the higher the value "

The value of a protocol node, v(_u_), is computed recursively based on the products of _u_. By introducing the evolutionary value of each node, the model captures the fact that the value of a protocol _u_ is driven by the values of the protocols that depend on it. For example, let’s consider again the internet protocol stack. TCP has a high evolutionary value because it is used by many higher-layer protocols -  some of them being valuable themselves. Let’s assume that we introduce a brand new protocol at the same layer as TCP that may have better performance or other great new features. The new protocol’s evolutionary value will be low if it is not used by important or popular higher-layer protocols, regardless of the great new features it may have. So the evolutionary value determines if the protocol will survive the competition with other protocols at the same layer that offers similar services.    

- **Node competitors and competition threshold:** We refer to the competitors of a node _u_, C(_u_), as the nodes at layer l that share at least a fraction c of node _u_’s products. We refer to the fraction c as the competition threshold. So, a node _w_ competes with a node _u_ if _w_ shares at least a fraction c of _u_’s products.

- **Node death rate:** The model has a death and birth process in place to account for the protocols that cease or get introduced, respectively. The competition among nodes becomes more intense, and it is more likely that a protocol _u_ dies if at least one of its competitors has a higher value than itself. When a node _u_ dies, then its products also die if their only substrate is _u_.

- **Node basic birth process:** The model, in its simplest version, has a basic birth process in place, where a new node is assigned randomly to a layer. The number of new nodes at a given time is set to a small fraction (say 1% to 10%) of the total number of nodes in the network at that time. So, the larger a protocol stack is, then the faster it grows.



#### **Toy example**

To illustrate the above model and the parameters, let’s consider a toy network example with L equal to 4 layers. The evolutionary value of each node is shown inside each circle. The generality probability for each layer is shown at the left of each layer, and it is denoted as s(l). As we noted earlier, the generality of the layers decreases as we move to higher layers, so on average, the number of products per node decreases as well. Let’s further assume that we have a competition threshold c = ⅗. Nodes _u_, _q_ and _w_ compete in layer 2. Nodes _u_ and _q_ compete, but this is unlikely to cause _q_ to die because _u_ and _q_ have comparable evolutionary values. In contrast, it is likely that _w_ will die because its value is much less than that of its maximum-value competitor, _u_.

![[Toy Network With 4 Layers.png]]

