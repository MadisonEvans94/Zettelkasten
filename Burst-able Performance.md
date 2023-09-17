#evergreen1 
upstream: [[EC2]]

---

**video links**: 

---


## What is Burst-able Performance?

**Burst-able performance** instances, provided by Amazon EC2, are designed to deliver a baseline level of CPU performance with the ability to burst to a higher level when required by your workload. These instances are ideal for workloads that don't need the full CPU continuously, like **micro-services**, **low-latency interactive applications**, **small and medium databases**, and **product prototypes**. 

This is achieved by using "**CPU Credits**". Each burstable instance continuously earns (at a millisecond-level resolution) a set rate of CPU credits per hour, depending on the instance size. 

Instances can burst above the baseline level using the CPU credits that they have accrued. If an instance does not use the credits it is earning, they will be stored in the CPU credit balance for up to one day for T2 instances, and indefinitely for T3/T3a instances

### ELI5: CPU credits: 

let's imagine CPU credits as **tokens in an arcade**!

So, when you go to an arcade, you're given tokens to play games. You can play as many games as you want as long as you have tokens. Now, imagine you get a few tokens every hour just for being in the arcade. These are like your "free" tokens. You can spend them on any game you like!

But let's say you find a really cool game, and you want to play it a lot. If you play it too much, you might run out of your tokens. If you run out of tokens, you can't play the game as much until you get more tokens.

>This is pretty similar to how CPU credits work on Amazon EC2

Your instance (like you in the arcade) gets a certain number of CPU credits (tokens) every hour. When your instance is doing easy tasks (like you playing a small game), it doesn't need a lot of CPU credits. But for harder tasks (like the really cool game), your instance needs to use more CPU credits. If your instance runs out of CPU credits, it has to slow down and can't do as much work until it earns more credits.

And just like how some arcades have rules about how many tokens you can save up, EC2 instances also have rules about how many CPU credits they can save up for later. The good thing is that if your instance is doing easy tasks and not using a lot of CPU credits, it can save them up for when it needs to do harder work!

## When to Use Burst-able Performance Instances?

Burst-able performance instances are intended for workloads that don't use the full CPU often or consistently, but occasionally need to burst (use more CPU). These instances are designed to provide a generous amount of CPU performance most of the time and the ability to burst to higher performance as required by the workload.

Some examples of appropriate workloads for burst-able performance instances are:

### Micro-services: 
Smaller services that don't require full CPU utilization but might need to burst during high traffic.

### Small and mid-size databases: 
Databases that usually operate with low CPU utilization and require bursts during heavy read/write operations.

### Low traffic websites or blogs:
Websites that have relatively low traffic most of the time but experience spikes in traffic occasionally.

### Development environments: 
These often have sporadic CPU usage, where high capacity is needed during active development, but not at other times.

### Build servers: 
Servers that are idle much of the time, but need to compile (build) software occasionally.

>When choosing an instance, it is always crucial to consider the nature of your workload, the need for speed, and cost-effectiveness. If your workload does not consistently need high levels of CPU, then burst-able performance instances can provide the performance you need with potentially significant cost savings.

For more details, always refer to the [AWS EC2 documentation](https://aws.amazon.com/ec2/instance-types/).






