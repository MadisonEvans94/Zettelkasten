#seed 
upstream: [[Django]]

---

**links**: 

---

Brain Dump: 

--- 



In Django REST Framework (DRF), how a viewset determines which method to run for a given HTTP request is governed by a combination of routing and the internal logic of the viewsets. This is based on both the HTTP method (GET, POST, PUT, DELETE, etc.) and the URL pattern.

### Understanding ViewSet Method Routing:

1. **HTTP Methods and Corresponding Actions**: DRF viewsets are designed to automatically map HTTP methods to specific actions. Here's a basic mapping:
   - `GET` (for a list): Maps to the `list` method.
   - `GET` (for a single item): Maps to the `retrieve` method.
   - `POST`: Maps to the `create` method.
   - `PUT`/`PATCH`: Maps to the `update` method.
   - `DELETE`: Maps to the `destroy` method.

2. **Router and URL Patterns**: When you use a router and register a viewset with it, DRF automatically creates URL patterns for these actions. For example, a `DefaultRouter` will create:
   - A URL pattern for `GET` requests to `/[prefix]/` that maps to the `list` method.
   - A URL pattern for `GET` requests to `/[prefix]/{id}/` that maps to the `retrieve` method.
   - A URL pattern for `POST` requests to `/[prefix]/` that maps to the `create` method.
   - ...and so on for other methods.

3. **Custom Methods**: If you add custom methods to a viewset, you use decorators like `@action` to specify additional details. For example, to add a custom method that responds to a `GET` request, you might do something like:

   ```python
   from rest_framework.decorators import action

   class ThoughtViewSet(viewsets.ModelViewSet):
       # ... existing methods ...

       @action(detail=True, methods=['get'])
       def custom_method(self, request, pk=None):
           # Custom method logic here
   ```

   This `@action` decorator tells DRF to create a URL pattern for this method, which would look like `/thoughts/{id}/custom_method/` for a `ThoughtViewSet` registered under the prefix `thoughts`.

### How DRF Knows Which Method to Run:

When a request comes in, DRF looks at the URL and the HTTP method. It then matches these against the URL patterns generated by the router and the method mappings inside the viewset. Based on this match, it determines which method on the viewset to execute.

### Summary:

- Standard CRUD operations in a viewset are automatically linked to corresponding HTTP methods (like `GET` for retrieving and `POST` for creating).
- Custom methods can be added using the `@action` decorator, which allows you to specify the URL subpath and the HTTP methods they respond to.
- The router plays a crucial role in creating URL patterns that map these HTTP methods and paths to the correct viewset methods.

