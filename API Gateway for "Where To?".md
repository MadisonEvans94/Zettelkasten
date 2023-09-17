This document details the setup of the API Gateway for the "Where To?" application. The API Gateway serves as an interface between the client application and the AWS backend services, allowing interactions with the AWS Lambda functions to perform operations such as creating and managing user groups, locations, and attributes.

> Keep in mind, nothing is set in stone, this is just potential at the moment 

## Overview

The API Gateway follows RESTful design principles. Each endpoint signifies a specific resource (**user groups**, **locations**, and **attributes**), while the HTTP method represents the operation to be performed on that resource. The hierarchy of the resources is mirrored in the endpoint structure.

## Endpoints

The following is the endpoint breakdown:

### `POST /usergroup`
To create a new user group, it maps to the `ManageUserGroupFunction`.

### `PUT /usergroup/:groupId/user/:userId`
To add a user to a group, it also maps to the `ManageUserGroupFunction`.

### `GET /usergroup/:groupId`
To fetch all information about a user group, including locations and attributes, it maps to a function that retrieves this data.

### `POST /usergroup/:groupId/location`
To create a new location within a group, it maps to the `ManageLocationFunction`.

### `PUT /usergroup/:groupId/location/:locationId`
To edit a location within a group, it maps to the `ManageLocationFunction`.

### `DELETE /usergroup/:groupId/location/:locationId`
To delete a location within a group, it maps to the `ManageLocationFunction`.

### `GET /usergroup/:groupId/location/:locationId`
To fetch information about a specific location within a group, it maps to a function that retrieves this data.

### `POST /usergroup/:groupId/location/:locationId/attribute`
To create a new attribute for a location, it maps to the `ManageAttributeFunction`.

### `PUT /usergroup/:groupId/location/:locationId/attribute/:attributeId`
To edit an attribute for a location, it maps to the `ManageAttributeFunction`.

### `DELETE /usergroup/:groupId/location/:locationId/attribute/:attributeId`
To delete an attribute for a location, it maps to the `ManageAttributeFunction`.

### `GET /usergroup/:groupId/location/:locationId/attribute/:attributeId`
To fetch information about a specific attribute for a location, it maps to a function that retrieves this data.

## Permissions

The API Gateway requires the `lambda:InvokeFunction` permission for the corresponding Lambda functions. This permission can be granted through an IAM role.

## Authorization

Authorization for the API Gateway can be facilitated through Amazon Cognito User Pools. The general procedure includes:

1. Establish a user pool in Amazon Cognito.
2. For each method, set the Authorization to the Amazon Cognito user pool in the API Gateway console.
3. Insert the JWT ID token from Cognito in the `Authorization` header when making calls to the API Gateway from the client application.

## Example

Below is an example of how to use the `POST /usergroup` endpoint:

### Request

```http
POST /usergroup HTTP/1.1
Content-Type: application/json
Authorization: Bearer <JWT token>

{
  "userId": "user1",
  "groupName": "group1"
}
```

### Request

```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "statusCode": 200,
  "body": "User group 'group1' successfully created."
}

```
