#seed 
upstream:

---

**links**: 

---
## Overview 

The software process goes as follows: 

- **analysis** 
- **specification**
- **design** 
- **implementation**

Overview of the methodology: Data First 
1. **Analysis: Information Flow Diagram** 
2. **Specification** 
	1. **ER Diagram** 
	2. **Tasks** 
3. **Design** 
	1. **Abstract Code w/SQL** 
	2. **Relation Schema**
4. **Implementation** 
	1. **Server-side code w/sql**
	2. **Database engine Integration (MySQL, PostgreSQL, etc)**



## Information Flow Diagrams
This is typically the first stage that you take in database design. This step has everything to do with gathering all of the customer requirements and organizing them into a flow chart 
![[Information Flow Diagram.png]]

- **Ellipses** represent tasks 
- **Rectangles** represents document 
- **Arrows** represent information flow. Direction of the error shows what direction the information flows (Read vs Write)
- **Dotted Lines** represent the system boundary (you can think of the dotted line as where the api will sit)

## Enhanced Entity-Relationship (EER) Diagram 

Let's go into the symbols...

### Entity type (rectangle)
- Normal (Default)
- Union Entity Types (think.. *this can either inherit from one entity or another. can't inherit from both*)

### property types (ellipse)
- Identifying properties (represented by an underlined label). This is a property that is unique to the entity (1 to 1 relationship)
- composite property type (a property that is made up of 2 or more other properties i/e: Name --> first name, last name) 
- multi-valued properties (represented by double ellipse). this is like having a one to many field 

###  Relationship Types 
(diamonds with connecting lines where each line has a number over it to indicate cardinality)
- 1-1 relationship 
- 1-many: it is a partial function which means that the *many's* of the model don't *have* to have a relationship with the *1's* 
- Mandatory 1-N Relationship: the *many* in the relationship must have a link to a *1* in the relationship or else it doesn't exist 
- Many to many 
- N-ary relationship types (i/e tertiay)
- Identifying Relationships/ Weak Entity Types (double lined diamond). Basically when you need specific properties to be present within each of the entities in the relationship in order to maintain the relationship. (think about users posting status updates. Each status needs to have a specific date and time and the user needs to have a specific property like email... so basically just indicating the need for primary keys of some sort)
- Recursive Relationship type (i/e: A user can be a supervisor for another user and so on)
- "is-a" relationship types or supertypes and subtypes (define the difference between `o` and `d`)

[[Mandatory Relationship Types.png]]
[[1 to Many Relationship Types.png]] 
[[Supertypes and Subtypes.png]]