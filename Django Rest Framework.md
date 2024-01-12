#seed 
upstream: [[Django]]

---

**links**: 

---

Brain Dump: 

--- 

## Installation 

Install using `pip`, including any optional packages you want...

```
pip install djangorestframework
pip install markdown       # Markdown support for the browsable API.
pip install django-filter  # Filtering support
```

...or clone the project from github.

```
git clone https://github.com/encode/django-rest-framework
```

Add `'rest_framework'` to your `INSTALLED_APPS` setting.

```
INSTALLED_APPS = [
    ...
    'rest_framework',
]
```

If you're intending to use the browsable API you'll probably also want to add REST framework's login and logout views. Add the following to your root `urls.py` file.

```
urlpatterns = [
    ...
    path('api-auth/', include('rest_framework.urls'))
]
```

Note that the URL path can be whatever you want.

---

## [Example](https://www.django-rest-framework.org/#example)

Let's take a look at a quick example of using REST framework to build a simple model-backed API.

We'll create a read-write API for accessing information on the users of our project.

Any global settings for a REST framework API are kept in a single configuration dictionary named `REST_FRAMEWORK`. Start off by adding the following to your `settings.py` module:

```python
REST_FRAMEWORK = {
    # Use Django's standard `django.contrib.auth` permissions,
    # or allow read-only access for unauthenticated users.
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.DjangoModelPermissionsOrAnonReadOnly'
    ]
}
```

Don't forget to make sure you've also added `rest_framework` to your `INSTALLED_APPS`.

We're ready to create our API now. Here's our project's root `urls.py` module:

```python
from django.urls import path, include
from django.contrib.auth.models import User
from rest_framework import routers, serializers, viewsets

# Serializers define the API representation.
class UserSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = User
        fields = ['url', 'username', 'email', 'is_staff']

# ViewSets define the view behavior.
class UserViewSet(viewsets.ModelViewSet):
    queryset = User.objects.all()
    serializer_class = UserSerializer

# Routers provide an easy way of automatically determining the URL conf.
router = routers.DefaultRouter()
router.register(r'users', UserViewSet)

# Wire up our API using automatic URL routing.
# Additionally, we include login URLs for the browsable API.
urlpatterns = [
    path('', include(router.urls)),
    path('api-auth/', include('rest_framework.urls', namespace='rest_framework'))
]
```

You can now open the API in your browser at [http://127.0.0.1:8000/](http://127.0.0.1:8000/), and view your new 'users' API. If you use the login control in the top right corner you'll also be able to add, create and delete users from the system.


---

## Understanding Serializers in Django Rest Framework

### What are Serializers?

Serializers in Django Rest Framework are responsible for converting complex data types, like querysets and model instances, into Python data types that can then be easily rendered into JSON, XML, or other content types. They also provide deserialization, allowing parsed data to be converted back into complex types, after validating the incoming data.

### Why Do We Need Serializers?

1. **Data Conversion**: Serializers handle the conversion of complex data to and from Python primitives. This is essential for creating RESTful APIs, where data needs to be sent or received in a format (like JSON) that's easily consumable by different clients.

2. **Data Validation**: They play a crucial role in validating data before it's saved to the database. DRF serializers come with built-in validation but can also be customized for more complex scenarios.

3. **Simplification**: By using serializers, we can abstract away the nitty-gritty details of how data is processed, making our code cleaner and more maintainable.

### How are they used in Django?

Here's how serializers are typically used in a Django project:

1. **Defining a Serializer**: Similar to Django forms, serializers define a set of fields. These fields determine how data will be serialized/deserialized.

    ```python
    from rest_framework import serializers
    from myapp.models import MyModel

    class MyModelSerializer(serializers.ModelSerializer):
        class Meta:
            model = MyModel
            fields = ['id', 'title', 'description']
    ```

2. **Serialization**: Convert model instances/querysets into Python data types. This typically happens in a view or viewset.

    ```python
    instance = MyModel.objects.get(id=1)
    serializer = MyModelSerializer(instance)
    serialized_data = serializer.data
    ```

3. **Deserialization and Validation**: Process incoming data, validate it, and then convert it into a complex type (like a model instance).

    ```python
    serializer = MyModelSerializer(data=request.data)
    if serializer.is_valid():
        instance = serializer.save()
    ```

4. **Customizing Serializer Behavior**: You can override methods like `create`, `update`, or validation methods to customize the serialization/deserialization process.

### Best Practices

- **Use ModelSerializers When Possible**: They provide a shortcut for creating serializers that deal with model instances and querysets.

- **Field-Level Validation**: Use field-level validation for simple checks and override `validate` method for object-level validation.

- **Read-Only Fields**: Mark fields as read-only if they should not be included in deserialization.

- **Nested Serialization**: For complex data structures involving relationships, nested serializers can be used.

- **Performance Considerations**: Be mindful of large querysets and consider using pagination or optimizing queries for performance.

### Key Takeaways

- Serialization is the process of transforming complex data (like Django model instances) into simpler, universally understandable formats (like JSON).
- It involves a field-by-field translation, where each model field is mapped to a corresponding serializer field, and the data is converted to basic Python data types.
- The end result is a format that can be easily transmitted over a network or stored in a format like JSON, XML, etc.



