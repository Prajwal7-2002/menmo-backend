# rag/models.py
import uuid
from django.db import models
from django.conf import settings

UserModel = settings.AUTH_USER_MODEL


class UploadedDocument(models.Model):
    """
    Metadata for uploaded documents.
    NOT required for Pinecone to work, but useful for debugging / admin.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        UserModel,
        on_delete=models.CASCADE,
        related_name="uploaded_documents",
    )
    title = models.CharField(max_length=512, blank=True)
    original_filename = models.CharField(max_length=512, blank=True)
    domain = models.CharField(max_length=256, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title or self.original_filename or str(self.id)


class DocumentChunk(models.Model):
    """
    Chunk records linked to UploadedDocument.
    Pinecone holds the vector + metadata;
    this is just a mirror for visibility.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    document = models.ForeignKey(
        UploadedDocument,
        on_delete=models.CASCADE,
        related_name="chunks",
    )
    chunk_idx = models.IntegerField()
    section_idx = models.IntegerField(null=True, blank=True)
    section_title = models.CharField(max_length=512, blank=True)
    page_num = models.IntegerField(null=True, blank=True)
    snippet = models.TextField(blank=True)
    pinecone_id = models.CharField(max_length=128, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["document", "chunk_idx"]

    def __str__(self):
        return f"{self.document_id} / chunk {self.chunk_idx}"

class ChunkFeedback(models.Model):
    """
    Aggregated feedback per Pinecone vector (chunk).
    Used to slightly boost or penalize chunks during retrieval.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    pinecone_id = models.CharField(max_length=128, unique=True)
    upvotes = models.IntegerField(default=0)
    downvotes = models.IntegerField(default=0)
    score = models.FloatField(default=0.0)  # up - down, can be weighted later
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-score"]

    def __str__(self):
        return f"{self.pinecone_id} (score={self.score})"

    @classmethod
    def apply_feedback(cls, pinecone_id: str, value: str) -> "ChunkFeedback":
        """
        Update aggregate feedback for a given chunk.
        value: "up" or "down".
        """
        if not pinecone_id:
            return None  # safety

        obj, _ = cls.objects.get_or_create(pinecone_id=pinecone_id)
        if value == "up":
            obj.upvotes += 1
            obj.score += 1.0
        elif value == "down":
            obj.downvotes += 1
            obj.score -= 1.0
        obj.save()
        return obj

class Conversation(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    title = models.CharField(max_length=255, default="New Chat")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]

    def __str__(self):
        return f"{self.title} ({self.id})"


class Message(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    conversation = models.ForeignKey(
        Conversation,
        on_delete=models.CASCADE,
        related_name="messages"
    )
    role = models.CharField(max_length=10)  # "user" / "assistant"
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]

    def __str__(self):
        return f"{self.role}: {self.content[:40]}"
