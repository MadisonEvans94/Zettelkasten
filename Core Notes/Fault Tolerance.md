#incubator 

###### upstream: [[Cloud Computing and Distributed Systems]]

### Analogy: 

Imagine you're on a field trip with your school. Your teacher has a map and is leading the way. But what happens if the teacher gets lost, or their map gets ruined? It's going to be a big problem, right?

But what if every student also had a copy of the map and knew the path? Even if the teacher got lost or the original map was ruined, any student could step up and lead the way. The field trip wouldn't be ruined, because the group has a backup plan.

This is a bit like how "fault tolerance" works in computing. A system is considered "fault tolerant" if it can keep working even when something goes wrong, like a component failing or a sudden surge in users.

### Key Terms: 

1.  **Redundancy**: This is like every student having their own map. In computing, it means having backup components or systems that can take over if the primary one fails.
    
2.  **Failover**: This is the process of switching to a redundant component or system when the primary one fails, like a student taking over if the teacher gets lost.
    
3.  **Recovery Point Objective (RPO)**: This is the maximum amount of data you can afford to lose if a failure happens. For example, if you back up your data every hour, your RPO would be one hour, because in the worst case, you could lose up to an hour's worth of data.
    
4.  **Recovery Time Objective (RTO)**: This is the maximum amount of time you can afford to be without the system after a failure. This depends on how quickly you can switch to a redundant component or system and get it up and running.
    
5.  **High Availability**: This is a characteristic of a system that's designed to be available as much as possible, minimizing the time it's down even in the event of a failure. High availability systems often use redundancy and failover to achieve this.
    
6.  **Fault Tolerant vs. Fault Resistant**: A system that is fault tolerant is designed to operate as expected in the event of a fault or failure. On the other hand, a system that is fault resistant is designed to avoid faults or failures in the first place. While these two concepts are related, they approach the problem of system failures from different angles.
    

Remember, the main goal of fault tolerance is to prevent disruptions and keep systems running smoothly for their users, even when things go wrong behind the scenes.


