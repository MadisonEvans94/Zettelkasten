#seed 
upstream: [[Databases]]

---

**links**: 

---





## Simple Case 

![[Screenshot 2024-03-01 at 8.37.47 AM.png]]
If an entity type ET has a single value property type B, then the relation ET will have B as an attribute

![[Screenshot 2024-03-01 at 8.39.13 AM.png]]
If ann= entity type ET has a property type A and that property type is an identifying property, then in the relation ET corresponding to it, there will be an attribute A which is a primary key/identifying attribute

![[Screenshot 2024-03-01 at 8.42.16 AM.png]]
If we have an entity type ET with a composite property type C, and that composite property type c is composed of D and E, then in the relation corresponding to ET, that adds two attributes D and E. But what happened to property type C? Well, property type C gets lost. C would be a redundant attribute in this case 

## Mapping Multivalued Attributes 
![[Screenshot 2024-03-01 at 8.48.42 AM.png]]
The notes you've provided on the mapping of multivalued attributes in Entity-Relationship (ER) diagrams into a relational database schema do make sense. Here's a clearer summary:

When you have an entity type (let's call it ET) with a unique identifying attribute (A), A serves as the primary key for ET in the relational database schema. If ET has a multivalued attribute (F), this cannot be directly represented in a single table due to the constraints of the relational model, which only allows atomic (single-valued) attributes.

To handle this, you create a new table specifically for managing the multivalued attribute. This table is named using the convention: entity name followed by the multivalued attribute name, separated by a hyphen (ET-F). The new table ET-F will have two columns: attribute A, which is a foreign key linking back to the original ET table, and attribute F.

The combination of A and F together becomes a composite key for the ET-F table. This is because each instance of ET can have multiple values for F, but each combination of A and F must be unique. In other words, for each A in the original ET table, there can be multiple Fs in the ET-F table, but each A-F pair must be unique.

The A attribute in the ET-F table serves two purposes: it ensures referential integrity by linking back to the ET table and helps maintain uniqueness for each F associated with an A.

To summarize in simpler terms:

1. Your entity ET has a single-valued primary key A.
2. Your entity ET also has a multivalued attribute F.
3. You can't store multivalued attributes directly in a relational table.
4. So, you create a new table ET-F to store these multivalued attributes.
5. In table ET-F, you store A (as a foreign key) and F, and together they form a unique composite key.
6. This allows ET to be associated with multiple Fs, but each A-F combination must be unique.
7. The A in table ET-F refers back to the primary key A in table ET, ensuring that F values are associated with the correct instance of ET.
![[Screenshot 2024-03-01 at 8.57.46 AM.png]]
## Mapping 1 to 1 relationships 
TLDR, if we have a simple 1 to 1 relationship, we can define a relationship by taking an entity ET1, insert its attributes, and then include the primary attribute of the entity that it has a 1 to 1 relationship with (in this case property B of ET2). This is a two way relationship. So valid solution would be either ET1 with attribute A and B as a foreign key, or ET2 with attribute B and A as a foreign key.
![[Screenshot 2024-03-01 at 9.03.50 AM.png]]

There is one particular instance where one solution is preferred over the other. This situation is if we have a mandatory relationship type. For example, if ET2 has a mandatory relationship (denoted by the double line connection to R) where every instance of ET2 **must** be related by relationship R to ET1. In other words, there cannot be an instance of ET2 that is not in a relationship with ET1. In that particular case, it is not advisable to use the first solution
![[Screenshot 2024-03-01 at 9.06.39 AM.png]]
## Mapping 1 to many relationships 
For 1-N relationships, we want to map the relationship such that the entity on the *N* side of the relationship has the primary attribute of the single entity in order to have a unique identifier. So the solution would be to describe ET2 with its attributes + the primary attribute of the single entity, which can be viewed as its foreign key 
![[Screenshot 2024-03-01 at 9.10.01 AM.png]]
## Mapping Many to Many Relationships 
Mapping many to many relationships is a bit different because we cannot uniquely identify based on foreign keys alone like we did in the 1-N case. Instead, we have to map the relationship in terms of the relation *R*. 

This means that for an N-M relationship between ET1 and ET2, we will map such that the solution is a separate relation of R that is a combination of attribute A and B that serves as a key. This enforces a unique representation. 
![[Screenshot 2024-03-01 at 9.12.55 AM.png]]
## Mapping Identifying Relationships with Weak Entity Types 

A weak entity type needs the identifying property of the strong entity type in order to identify them. So the solution is to insert into ET2 a reference to ET1,  that is the A attribute becomes a foreign key for ET2

![[Screenshot 2024-03-01 at 9.31.08 AM.png]]
## Mapping Super-Sub type relationships 

### Case 1: Mandatory Disjoint

In a mandatory disjoint relationship, every single instance of ET needs to be hooked up to ET1 or ET2.

Every single ET entity will have an A value and a B value. But we also know that every single instance of ET is hooked up to either to ET1 or to ET2. Hence, ET1 and ET2 will each inherit the attributes of ET as well as tacking on its own specific attributes. 
![[Screenshot 2024-03-01 at 9.34.17 AM.png]]
Well what about the pure ET entities? Well, that cannot happen because we have defined this as *mandatory*, meaning that the super type cannot exist on its own. (Think of this like an abstract class from which child classes inherit from)
### Case 2: Mandatory Overlapping 

A slight variation is mandatory overlapping. In other words, every single instance of ET is ET1, or ET2, or **both**. 
![[Screenshot 2024-03-01 at 9.38.12 AM.png]]
### Case 3: Non-Mandatory Overlap 

![[Screenshot 2024-03-01 at 9.40.06 AM.png]]

### Case 4: Non-Mandatory Disjoint 
- [ ] review non-mandatory vs mandatory meaning 
![[Screenshot 2024-03-01 at 9.41.13 AM.png]]
### Case 5: Union Type 
- [ ] review union types 
![[Screenshot 2024-03-01 at 9.42.45 AM.png]]

