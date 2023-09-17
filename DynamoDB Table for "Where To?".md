This is a notes section for the database in the backend of the Where To app

---

## Application ERD 

![[WhereToERD.png]]

---

### User 
Represents an individual user in the system.

**ERD Representation**:

![[Screen Shot 2023-08-08 at 1.18.28 PM.png]]

**Access Patterns**: 
- Retrieve a User's Details

> When a user logs into the application, their profile details are fetched to be displayed on their dashboard.

**Lambda**: 
*getUserDetails()*

**Example JSON Object**:
```json
{
  "PK": "USER#00001",
  "SK": "#METADATA#00001",
  "userName": "MadisonEvans94"
}
```

---

### Group
Represents a group created by a user.

**ERD Representation**:

![[Screen Shot 2023-08-08 at 1.15.44 PM.png]]

**Access Patterns**: 
- Get details of a specific group

> When navigating to a specific group's page, the group's details, such as its name and admin, are displayed at the top.

**Lambda**: 
*getGroupDetails()*

**Example JSON Object**:
```json
{
  "PK": "GROUP#000001",
  "SK": "#METADATA#0000001",
  "groupName": "TravelBuddies",
  "adminUserID": "11947"
}
```

---

### GroupMembership
Shows all groups of a specific user 

**ERD Representation**:

![[Screen Shot 2023-08-08 at 1.16.43 PM.png]]

**Access Pattern**: 
- List all groups a user is a part of

> After logging in, the user's dashboard displays a list of all the groups they are a part of, allowing them to select one to view or interact with.

**Lambda Function**: 
*getMemberships()*

**Example JSON Object**:
```json
{
  "PK": "USER#U1",
  "SK": "GROUP#G1"
}
```

---

### GroupUser
Shows all users of a specific group

> Note: this is an inverse of the Membership entity and will be utilized like a backref

**ERD Representation**:

![[Screen Shot 2023-08-08 at 1.17.04 PM.png]]

**Access Patterns**: 
- List All Users in a Specific Group

> when you click on one of your groups, application will fetch to see who all is in that group, perhaps to view members or manage invitations.

**Lambda**: 
*getUsersOfGroup()*

**Example JSON Object**:
```json
{
  "PK": "USER#005",
  "SK": "GROUP#008"
}
```
---

### Location
Represents a user within a specific group

**ERD Representation**:
![[Screen Shot 2023-08-08 at 1.17.23 PM.png]]

**Access Pattern**: 
- Add a New Location to a Group with Scores

>Within a group, you decide to share your experience about a recent trip to Miami. You add Miami as a location and provide scores based on your experience

**Lambda**: 
*getLocation()*
*addLocation()*

**Example JSON Object**:
```json
{
  "PK": "GROUP#G1",
  "SK": "LOCATION#L1",
  "locationName": "Atlanta, GA",
  "scores": {
    "U1": {
      "transportation": 6,
      "cost of living": 6,
      "food": 6
    }
  },
  "averageScores": {
    "transportation": 6,
    "cost of living": 6,
    "food": 6
  }
}
```

---

### UserLocationScore
Represents a location added to a group with scores.

**ERD Representation**:

![[Screen Shot 2023-08-08 at 1.17.58 PM.png]]

**Access Pattern**: 
- List All Users Who Scored a Specific Location

> While viewing the scores for New York City in your group, you're curious about who has been there. You click on an option to "See all users who scored this location" and get a list of members who've shared their NYC experiences.

**Lambda**: 
*getLocation()*

**Example JSON Object**:
```json
{
  "PK": "USER#U1",
  "SK": "LOCATION#L1",
  "scores": {
    "transportation": 6,
    "cost of living": 6,
    "food": 6
  }
}
```

---
