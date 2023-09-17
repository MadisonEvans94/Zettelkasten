

This document outlines the `ManageLocationFunction` Lambda function for the "Where To?" application. This function is responsible for managing locations within user groups, including creation, modification, and deletion of locations, as well as modification of location attributes.

> Please note, the design outlined here is tentative and subject to change.

## Function Overview

The `ManageLocationFunction` handles the following operations:

1. Adding a new location to a user group in the UserGroups table in DynamoDB.
2. Modifying an existing location's details and attributes.
3. Removing a location from a user group.

The operation to execute is determined by the `action` parameter in the request.

## Expected Inputs

The function anticipates a JSON object with the following structure:

```json
{
  "action": "addLocation" | "editLocation" | "deleteLocation",
  "userId": "<userId>",
  "groupId": "<groupId>",
  "locationId": "<locationId>",
  "locationDetails": {
    "name": "<locationName>",
    "attribute1": "<attribute1Value>",
    "attribute2": "<attribute2Value>",
    ...
  }
}
```

The `action` key specifies the action to perform. It must be one of: `"addLocation"`, `"editLocation"`, or `"deleteLocation"`.

The `locationDetails` object contains the location's name and its fixed set of attributes. This object should be included when adding or editing a location.

## Expected Outputs

The function returns a JSON object with the following structure:

```json
{
  "statusCode": 200,
  "body": "<Response Message>"
}
```

The `statusCode` is a status code that reflects the outcome of the operation. A `200` status code indicates success.

The `body` is a message that describes the operation's result.

## Permissions

The `ManageLocationFunction` requires permissions to read from and write to the UserGroups table in DynamoDB. The necessary permissions can be assigned through an IAM role associated with the Lambda function. This role should include `dynamodb:GetItem`, `dynamodb:PutItem`, `dynamodb:UpdateItem`, and `dynamodb:DeleteItem` permissions for the UserGroups table.

## Example

Here is a sample request to add a new location:

```json
{
  "action": "addLocation",
  "userId": "user1",
  "groupId": "group1",
  "locationId": "location1",
  "locationDetails": {
    "name": "City2",
    "Climate": "Tropical",
    "Population": "1 Million",
    ...
  }
}
```

This request adds a new location "City2" with a set of attributes to the group with ID "group1".

The expected response could look something like:

```json 
{
  "statusCode": 200,
  "body": "Location 'City2' successfully added to group 'group1'."
}
```