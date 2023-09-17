Date: 08/03/2023

## Summary
A collaborative ranking application, where users within a private group can rate different locations based on their personal preferences (I guess kinda like yelp? But private and with data visualization elements). The application aims to provide a visual representation of user opinions and preferences, making the decision-making process easier and more fun for groups looking to relocate or travel.

## Objectives
- To create a personalized, visual representation of various locations based on the group's unique preferences
- To facilitate group decision-making by providing a centralized platform to share and rank opinions.
- To make location comparisons dynamic and adaptable by allowing users to assign and scale custom attributes.

| **User Story ID** | **User Type** | **User Story** | **Acceptance Criteria** |
| --- | --- | --- | --- |
| 1 | General User|As a user, I want to create a new private group, so that I can collaborate with specific people. |User is able to create a new group and invite others by email or username. |
| 2 | General User|As a user, I want to add a city to our group's list, so that it can be evaluated and ranked. |User can search for a city and add it to the group's list. |
| 3 | General User|As a user, I want to add positive or negative attributes to a city, so that the city can be rated according to our group's preferences. | User can add an attribute to a city and assign it a positive or negative value.|
| 4 | General User| As a user, I want to scale attributes based on importance, so that the overall score reflects our priorities.| User can adjust the scale of attributes for a city, influencing the overall score.|
| 5 | General User| As a user, I want to see a visual representation of our group's city rankings, so that we can easily compare cities.| User sees a visual (e.g., a map with dots of varying sizes) representing the net sum of attribute points for each city.|
| 6 | General User| As a user, I want to edit or delete attributes I've added, so that I can correct errors or change my opinion.| User can edit or remove attributes they've added to a city.|

## Random Thoughts 

- one challenge I see is whether or not to have attributes set in stone from the jump, or have them be completely open to interpretation. If we go with the later, then I see Mongo/DynamoDB as being an easy solution for the main database used. The only issue is that over time, each entry could end up having a shit ton of attributes that are null... for example, if `user1` adds something like *"this place has good hot dogs... +1"*, then each location in the table will automatically have *good hot dogs* as an additional attribute, set to null by default (if I'm wrong, correct me on that). But on the other hand, what if we hard code the attributes of a place that we can add a score to (such as walkability, cost of living, how's the weather, entertainment, etc...) and then we add someone to the group who really values an attribute such as diversity and inclusion or sports teams... would we be loosing valuable data by not allowing flexibility in the structure? 

 > Look into Mercer's quality of living index
 
 
- Do we want to introduce any level of democracy to this? Or do we want to let the scores balance naturally? If the collaboration group is small and everyone is pretty like minded, then that's kind of a non-issue. But let's say we have a huge group of users in a collaboration space, all of which are adding their own attributes and scores to different places. Do we need to have a way to vote? Maybe not on actions like adding a score to an attribute, but if we for example decide to go with a NoSQL db that appends a new attribute to every element in a table, should we require the group to vote on it? (for example, if I want to add something like *"how nice are the people"* as an attribute for **every** location, should I be able to do that without getting a sign off from everyone else in my group?)

## Potential Ideas on Architecture 

> Note: This is just me off the dome spit balling here.. so nothing set in stone. take with a grain of salt 

![[IMG_C36FC9A5E78A-1.jpeg]]

## What it could look like...

![[IMG_8308FC95506A-1.jpeg]]

- AWS **Madison, George**
	- Database + CRUD and Lambda 
	- UserPools and Auth 

- Frontend & UI design **Gherig**, **Justin?**
	- Data visualization (D3)
	- 3 js 
	- Components and Responsive design 
