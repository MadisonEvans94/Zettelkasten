#seed 
upstream: [[Django]]

---

**links**: [tutorial](https://www.youtube.com/watch?v=llrIu4Qsl7c&t=204s&ab_channel=AdamLaMorre)

---

Brain Dump: 
- get_object_or_404
- from rest_framework.authtoken.models import Token
- from django.contrib.auth.models import User

--- 

```python
from django.shortcuts import render
from rest_framework.decorators import api_view, authentication_classes, permission_classes
from rest_framework.response import Response
from .serializers import UserSerializer
from django.contrib.auth.models import User
from rest_framework import status
from rest_framework.authtoken.models import Token
from rest_framework.authentication import SessionAuthentication, TokenAuthentication
from rest_framework.permissions import IsAuthenticated
from rest_framework.exceptions import AuthenticationFailed
from rest_framework.throttling import AnonRateThrottle

@api_view(['POST'])
def signup(request):
    serializer = UserSerializer(data=request.data)
    if serializer.is_valid(raise_exception=True):
        user = serializer.save()
        user.set_password(request.data['password'])
        user.save()
        token = Token.objects.create(user=user)
        return Response({"token": token.key, "user": serializer.data}, status=status.HTTP_201_CREATED)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['POST'])
@authentication_classes([])
@permission_classes([])
def login(request):
    try:
        user = User.objects.get(username=request.data.get('username'))
    except User.DoesNotExist:
        raise AuthenticationFailed('Invalid credentials')

    if not user.check_password(request.data.get('password')):
        raise AuthenticationFailed('Invalid credentials')

    token, created = Token.objects.get_or_create(user=user)
    serializer = UserSerializer(instance=user)
    return Response({"token": token.key, "user": serializer.data}, status=status.HTTP_200_OK)

@api_view(['POST'])
@authentication_classes([TokenAuthentication])
@permission_classes([IsAuthenticated])
def logout(request):
    # Delete the token to log the user out
    request.user.auth_token.delete()
    return Response({"message": "Successfully logged out."}, status=status.HTTP_200_OK)


@api_view(['GET'])
@authentication_classes([SessionAuthentication, TokenAuthentication])
@permission_classes([IsAuthenticated])
def test_token(request):
    return Response({"message": f"Token valid for {request.user.email}"}, status=status.HTTP_200_OK)

```

---

## Overview
This document explains the Django authentication system implemented in the provided code. The system uses Django Rest Framework (DRF) for creating RESTful APIs for user authentication, including signup, login, and logout functionalities.

## Key Components

### Decorators
- `@api_view`: This decorator is used to restrict the allowed HTTP methods for a view. For example, `@api_view(['POST'])` means the view only responds to POST requests.
  
- `@authentication_classes` and `@permission_classes`: These decorators are used to specify the authentication and permission policies for a view. For instance, `@authentication_classes([TokenAuthentication])` applies token-based authentication.

### Classes
- `User`: A model provided by Django's `django.contrib.auth.models`, representing the user of the system.

- `Token`: Part of DRF's token authentication system, this model stores a unique token for each user.

- `SessionAuthentication` and `TokenAuthentication`: Authentication classes from DRF. `SessionAuthentication` uses Django's session framework for authentication, while `TokenAuthentication` uses token-based authentication.

- `IsAuthenticated`: A permission class in DRF that grants access to a view only to authenticated users.

- `AuthenticationFailed`: An exception class used to signal authentication failures.

- `AnonRateThrottle`: A DRF class for implementing rate limiting. It restricts the rate at which anonymous users can make requests.

## Core Functions

### `signup(request)`
This function handles user registration.

- **Serializer**: `UserSerializer` is used to validate and serialize request data. If the data is valid, a new `User` instance is created and saved.

- **Password Handling**: The user's password is set using Django's built-in password management system to ensure security.

- **Token Creation**: A token is generated for the new user using `Token.objects.create(user=user)`.

- **Response**: On successful registration, the user's data and token are returned.

### `login(request)`
This function manages user login.

- **User Retrieval**: It attempts to retrieve the user based on the username provided. If the user does not exist, an `AuthenticationFailed` exception is raised.

- **Password Verification**: The user's password is verified. If incorrect, `AuthenticationFailed` is raised.

- **Token Handling**: A token is retrieved or created for the authenticated user.

- **Response**: The user's data and token are returned upon successful authentication.

### `logout(request)`
This function facilitates user logout.

- **Token Deletion**: The authenticated user's token is deleted using `request.user.auth_token.delete()`, effectively logging them out.

- **Response**: A successful logout message is sent.

### `test_token(request)`
This function serves as a test endpoint to verify token validity.

- **Authentication and Permissions**: It requires token authentication and checks if the user is authenticated.

- **Response**: A confirmation message with the user's email is sent if the token is valid.
