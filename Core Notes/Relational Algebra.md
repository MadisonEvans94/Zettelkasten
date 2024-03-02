#seed 
upstream:

---

**links**: 

braindump: 
- [ ] define "closed" query language 
- [ ] define what it means by "relations are sets" with examples 
---
## Relations are Sets 
TODO 
## Selection
$$
\sigma_{expression}(R)
$$
select all the tuples from relation $R$ that satisfy this $expression$

### Example 1

*Find all RegularUsers with HomeTown Atlanta*

**Expression:**
$$
\sigma_{HomeTown=Atlanta}(RegularUser)
$$
**RegularUser Table:**

| Email         | Birth Year | Sex | Current City | HomeTown |
| ------------- | ---------- | --- | ------------ | -------- |
| user2@gt.edu  | 1969       | M   | Austin       | Austin   |
| user3@gt.edu  | 1967       | M   | San Diego    | Portland |
| user9@gt.edu  | 1988       | F   | Las Vegas    | Atlanta  |
| user10@gt.edu | 1986       | M   | Dallas       | Dallas   |
| user12@gt.edu | 1974       | F   | College Park | Austin   |
**Result:**

| Email         | Birth Year | Sex | Current City | HomeTown |
| ------------- | ---------- | --- | ------------ | -------- |
| user9@gt.edu  | 1988       | F   | Las Vegas    | Atlanta  |
### Example 2

*Find all RegularUsers with the same CurrentCity and Hometown or with Hometown Atlanta*
**Expression:**
$$
\sigma_{CurrentCity=HomeTown\,OR\,HomeTown=Atlanta}(RegularUser)
$$
**Result**

| Email         | Birth Year | Sex | Current City | HomeTown |
| ------------- | ---------- | --- | ------------ | -------- |
| user2@gt.edu  | 1969       | M   | Austin       | Austin   |
| user9@gt.edu  | 1988       | F   | Las Vegas    | Atlanta  |
| user10@gt.edu | 1986       | M   | Dallas       | Dallas   |

## Projection 

$$
\pi_{A1,A2,...,An}(R)
$$
project from relation $R$ the attributes $A1$, $A2$, ... , $An$ where $A1$, $A2$, ... , $An$ are all attributes within $R$
### Example 1

*Find Email, BirthYear, and Sex for RegularUsers in Atlanta*

**Expression:**
$$
\pi_{Email,BirthYear,Sex}(\sigma_{HomeTown='Atlanta'}(RegularUser))
$$
**RegularUser Table:**

| Email         | Birth Year | Sex | Current City | HomeTown |
| ------------- | ---------- | --- | ------------ | -------- |
| user2@gt.edu  | 1969       | M   | Austin       | Austin   |
| user3@gt.edu  | 1967       | M   | San Diego    | Portland |
| user9@gt.edu  | 1988       | F   | Las Vegas    | Atlanta  |
| user10@gt.edu | 1986       | M   | Dallas       | Dallas   |
| user12@gt.edu | 1974       | F   | College Park | Austin   |
**Result:**

| Email        | Birth Year | Sex |
| ------------ | ---------- | --- |
| user9@gt.edu | 1988       | F   |


## Union-U

### Example 1

*Find all cities that are a CurrentCity or a Hometown for some RegularUser*

**Expression:**
$$
\pi_{CurrentCity}(RegularUser)\cup\pi_{HomeTown}(RegularUser)
$$

> Note that the two relations in a union, intersection or set difference must be type compatible, i.e. same number of attributes and pairwise same types 

**RegularUser Table:**

| Email         | Birth Year | Sex | Current City | HomeTown |
| ------------- | ---------- | --- | ------------ | -------- |
| user2@gt.edu  | 1969       | M   | Austin       | Austin   |
| user3@gt.edu  | 1967       | M   | San Diego    | Portland |
| user9@gt.edu  | 1988       | F   | Las Vegas    | Atlanta  |
| user10@gt.edu | 1986       | M   | Dallas       | Dallas   |
| user12@gt.edu | 1974       | F   | College Park | Austin   |
**Result:**

|              |
| ------------ |
| Austin       |
| San Diego    |
| Portland     |
| Atlanta      |
| Las Vegas    |
| College Park |
| Dallas       |


## Intersection 

### Example 1

*Find all cities that are a CurrentCity for someone and a Hometown for some RegularUser*

**Expression:**
$$
\pi_{CurrentCity}(RegularUser)\cap\pi_{HomeTown}(RegularUser)
$$

**RegularUser Table:**

| Email         | Birth Year | Sex | Current City | HomeTown |
| ------------- | ---------- | --- | ------------ | -------- |
| user2@gt.edu  | 1969       | M   | Austin       | Austin   |
| user3@gt.edu  | 1967       | M   | San Diego    | Portland |
| user9@gt.edu  | 1988       | F   | Las Vegas    | Atlanta  |
| user10@gt.edu | 1986       | M   | Dallas       | Dallas   |
| user12@gt.edu | 1974       | F   | College Park | Austin   |
**Result:**

|        |
| ------ |
| Austin |
| Dallas |

## Set Difference 

### Example 1

- [ ]  include a vin diagram image 

*Find all cities that are a CurrentCity for some RegularUser, but exclude those that are a HomeTown for some RegularUser*

> in other words... show me the cities that appear in Current City but not in HomeTown 

**Expression:**
$$
\pi_{CurrentCity}(RegularUser)\textbackslash\pi_{HomeTown}(RegularUser)
$$

**RegularUser Table:**

| Email         | Birth Year | Sex | Current City | HomeTown |
| ------------- | ---------- | --- | ------------ | -------- |
| user2@gt.edu  | 1969       | M   | Austin       | Austin   |
| user3@gt.edu  | 1967       | M   | San Diego    | Portland |
| user9@gt.edu  | 1988       | F   | Las Vegas    | Atlanta  |
| user10@gt.edu | 1986       | M   | Dallas       | Dallas   |
| user12@gt.edu | 1974       | F   | College Park | Austin   |
**Result:**

|              |
| ------------ |
| San Diego    |
| Las Vegas    |
| College Park |

## Natural Join 
### Example 

*Find Email, Year, Sex, and Event when the Birth Year of the RegularUser is the same as the Event Year of the Major60sEvents*

**Expression:**
- [ ]  figure out appropriate expression 

**RegularUser Table:**

| Email         | Year | Sex | Current City | HomeTown |
| ------------- | ---- | --- | ------------ | -------- |
| user2@gt.edu  | 1969 | M   | Austin       | Austin   |
| user3@gt.edu  | 1967 | M   | San Diego    | Portland |
| user9@gt.edu  | 1988 | F   | Las Vegas    | Atlanta  |
| user10@gt.edu | 1986 | M   | Dallas       | Dallas   |
| user12@gt.edu | 1974 | F   | College Park | Austin   |

**Major60sEvents Table:**

| Year | Event                |
| ---- | -------------------- |
| 1963 | March On Washington  |
| 1963 | Ich bin ein          |
| 1963 | JFK assassination    |
| 1962 | Cuban Missile Crisis |
| 1961 | Berlin Wall          |
| 1968 | Tet Offensive        |
| 1968 | Bloody Sunday        |
| 1968 | MLK assassination    |
| 1969 | Moon Landing         |
| 1967 | The Doors            |
| 1966 | Rolling Stones       |

**Result:**

| Email        | Year | Sex | Event        |     |
| ------------ | ---- | --- | ------------ | --- |
| user2@gt.edu | 1969 | M   | Moon Landing |     |
| user3@gt.edu | 1967 | M   | The Doors    |     |
|              |      |     |              |     |
### Properties 
- matches values of attributes with *same names*
- keeps only one copy of the join attribute(s)
- is an "inner" join, meaning that only the tuples that actually appear in the relation and match will appear in the result

## Theta Join
### Example 

*Find Email, Year, Sex, and EventYear when the BirthYear of the RegularUser is before the EventYear of the Major60sEvent*

**Expression:**
$$
RegularUser\bowtie_{BirthYear<EventYear}Major60sEvents
$$

**RegularUser Table:**

| Email         | Year | Sex | Current City | HomeTown |
| ------------- | ---- | --- | ------------ | -------- |
| user2@gt.edu  | 1969 | M   | Austin       | Austin   |
| user3@gt.edu  | 1967 | M   | San Diego    | Portland |
| user9@gt.edu  | 1988 | F   | Las Vegas    | Atlanta  |
| user10@gt.edu | 1986 | M   | Dallas       | Dallas   |
| user12@gt.edu | 1974 | F   | College Park | Austin   |

**Major60sEvents Table:**

| Year | Event                |
| ---- | -------------------- |
| 1963 | March On Washington  |
| 1963 | Ich bin ein          |
| 1963 | JFK assassination    |
| 1962 | Cuban Missile Crisis |
| 1961 | Berlin Wall          |
| 1968 | Tet Offensive        |
| 1968 | Bloody Sunday        |
| 1968 | MLK assassination    |
| 1969 | Moon Landing         |
| 1967 | The Doors            |
| 1966 | Rolling Stones       |

**Result:**
- [ ] show the correct results 

### Properties 
- $\theta$: comparison expression 
- all attributes are preserved 
- also an "inner" join 


## Left Outer Join
### Example 

*Find Email, Year, Sex, and Event when the Birth Year of the RegularUser is the same as the Event Year of the Major60sEvents*

**Expression:**
- [ ]  figure out appropriate expression 

**RegularUser Table:**

| Email         | Year | Sex | Current City | HomeTown |
| ------------- | ---- | --- | ------------ | -------- |
| user2@gt.edu  | 1969 | M   | Austin       | Austin   |
| user3@gt.edu  | 1967 | M   | San Diego    | Portland |
| user9@gt.edu  | 1988 | F   | Las Vegas    | Atlanta  |
| user10@gt.edu | 1986 | M   | Dallas       | Dallas   |
| user12@gt.edu | 1974 | F   | College Park | Austin   |

**Major60sEvents Table:**

| Year | Event                |
| ---- | -------------------- |
| 1963 | March On Washington  |
| 1963 | Ich bin ein          |
| 1963 | JFK assassination    |
| 1962 | Cuban Missile Crisis |
| 1961 | Berlin Wall          |
| 1968 | Tet Offensive        |
| 1968 | Bloody Sunday        |
| 1968 | MLK assassination    |
| 1969 | Moon Landing         |
| 1967 | The Doors            |
| 1966 | Rolling Stones       |

**Result:**
- [ ] figure out appropriate results table 
## Cartesian Product-X
## Divideby
## Rename 