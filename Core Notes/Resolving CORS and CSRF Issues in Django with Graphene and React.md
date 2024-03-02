#seed 
upstream: [[Django]], [[Graphene]]

---

**links**: 

---

Brain Dump: 

--- 

## Issue Overview

When integrating a React frontend using Apollo Client with a Django backend serving a Graphene GraphQL API, CORS (Cross-Origin Resource Sharing) and CSRF (Cross-Site Request Forgery) issues can arise, leading to errors such as "Forbidden (Origin checking failed)".

## Why It Occurs

1. **CORS Issues**: Occur due to security measures preventing web applications from making requests to a domain different from the one that served the web page.

2. **CSRF Issues**: Django's CSRF protection can interfere with state-changing requests (like POST requests in GraphQL) from a frontend application.

## Solution

### Exempting GraphQL Endpoint from CSRF Verification:

In Django's `urls.py`, exempt the GraphQL view from CSRF checks:

```python
from django.views.decorators.csrf import csrf_exempt
from graphene_django.views import GraphQLView

urlpatterns = [
    path('graphql/', csrf_exempt(GraphQLView.as_view(graphiql=True))),
    # other paths...
]
```

## Best Practices

1. **Use CSRF Protection Judiciously**: CSRF is vital for security. Exempt API endpoints (like GraphQL) if you're using token-based authentication, but ensure CSRF protection for other parts of the application.

2. **Consistent CORS Configuration**: Ensure CORS settings in Django (`django-cors-headers`) are consistent and allow requests from the React application's domain.

3. **Secure Production Environment**: Be more restrictive with CORS in production. Avoid using `CORS_ALLOW_ALL_ORIGINS = True` in production settings.

4. **Environment Variables**: Use environment variables to manage different CORS policies for development and production environments.

5. **Testing**: Test both locally and in your deployment environment to ensure configuration works as expected.
