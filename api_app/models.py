# api_app/models.py
import uuid
from django.db import models
from django.conf import settings

UserModel = settings.AUTH_USER_MODEL


class QueryLog(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(UserModel, on_delete=models.SET_NULL, null=True, blank=True)
    query = models.TextField()
    answer = models.TextField(blank=True)
    top_score = models.FloatField(default=0.0)
    chunks = models.JSONField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]


class Feedback(models.Model):
    CHOICES = (("up", "up"), ("down", "down"))

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    query_log = models.ForeignKey(QueryLog, on_delete=models.CASCADE, related_name="feedbacks")
    user = models.ForeignKey(UserModel, on_delete=models.SET_NULL, null=True, blank=True)
    value = models.CharField(max_length=10, choices=CHOICES)
    reason = models.CharField(max_length=255, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]


class UserPreference(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(UserModel, on_delete=models.CASCADE)
    key = models.CharField(max_length=100)
    value = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ("user", "key")


class ConversationSummary(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(UserModel, on_delete=models.CASCADE)
    summary = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

class Conversation(models.Model):
    id          = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user        = models.ForeignKey(UserModel, on_delete=models.CASCADE, related_name="conversations")
    title       = models.CharField(max_length=200, default="New Chat")
    created_at  = models.DateTimeField(auto_now_add=True)
    updated_at  = models.DateTimeField(auto_now=True)

class Message(models.Model):
    id            = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    conversation  = models.ForeignKey(Conversation, on_delete=models.CASCADE, related_name="messages")
    role          = models.CharField(max_length=20, choices=[("user","user"),("assistant","assistant")])
    content       = models.TextField()
    timestamp     = models.DateTimeField(auto_now_add=True)