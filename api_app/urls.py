from django.urls import path
from .views import (
    AskAPIView,
    FeedbackAPIView,
    AnalyticsAPIView,
    UploadDocumentAPIView,
    DocumentListAPIView,
    DocumentDeleteAPIView,
    SwitchDomainAPIView,
    CreateConversationAPIView,
    ConversationListAPIView,
    ConversationChatAPIView,
    ConversationHistoryAPIView,
    RenameConversationAPIView,
    DeleteConversationAPIView
)

urlpatterns = [
    path("ask/", AskAPIView.as_view(), name="ask"),
    path("feedback/", FeedbackAPIView.as_view(), name="feedback"),
    path("analytics/", AnalyticsAPIView.as_view(), name="analytics"),

    path("upload-document/", UploadDocumentAPIView.as_view(), name="upload_document"),
    path("my-documents/", DocumentListAPIView.as_view(), name="my_documents"),

    path("delete-document/<uuid:doc_id>/", DocumentDeleteAPIView.as_view(), name="delete_document"),
    path("switch-domain/", SwitchDomainAPIView.as_view(), name="switch_domain"),
    # 🧵 multi-chat
    path("conversations/", CreateConversationAPIView.as_view(), name="create_conversation"),
    path("conversations/list/", ConversationListAPIView.as_view(), name="list_conversations"),
    path("conversations/<uuid:cid>/chat/", ConversationChatAPIView.as_view(), name="conversation_chat"),
    path("conversations/<uuid:cid>/history/", ConversationHistoryAPIView.as_view(), name="conversation_history"),
    path("conversations/<uuid:cid>/rename/", RenameConversationAPIView.as_view()),       # rename
    path("conversations/<uuid:cid>/delete/", DeleteConversationAPIView.as_view()),       # delete
]
