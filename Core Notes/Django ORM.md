#seed 
upstream:

---

**links**: 

---

Brain Dump: 

--- 



In the Django `models.py`, each model corresponds to a table in your database, and each attribute of the model corresponds to a field in the table. The `ForeignKey` is used to define a many-to-one relationship, where the `on_delete=models.CASCADE` parameter means that if the referenced object is deleted, also delete the objects that have a ForeignKey to it.

Remember that Django's `User` model has a lot of built-in functionality, including password hashing and reset functionality, so in a real-world scenario, it might be beneficial to extend the built-in `User` model rather than creating a completely custom one. Also, Django's `User` model already includes a `username` and `email` field, so you could simply extend it with a one-to-one link to a profile model to store additional user information.



![[HEADSPACE_ERD.pdf]]

```python 
from django.db import models
from django.contrib.auth.models import User
from django.contrib.postgres.fields import ArrayField
import uuid

# Extend the built-in User model with a One-To-One link to a new model that stores additional information
class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    registration_date = models.DateTimeField(auto_now_add=True)
    head_space = models.ForeignKey('HeadSpace', on_delete=models.CASCADE)

    def __str__(self):
        return self.user.username

class HeadSpace(models.Model):
    head_space_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    settings = models.JSONField()  # Storing settings as JSON

    def __str__(self):
        return str(self.head_space_id)

class Thought(models.Model):
    thought_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    content = models.CharField(max_length=2000)
    post_date = models.DateTimeField()
    # Use ArrayField for PostgreSQL; otherwise, you may serialize to a TextField.
    embedding_vector = ArrayField(models.FloatField(), size=your_fixed_length)
    cluster = models.ForeignKey('Cluster', on_delete=models.SET_NULL, null=True, blank=True)

    def __str__(self):
        return self.content[:50]

class Cluster(models.Model):
    cluster_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    embedding_value = ArrayField(models.FloatField(), size=your_fixed_length)
    description = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name


```