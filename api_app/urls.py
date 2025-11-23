from django.urls import path
from .views import (
    AskAPIView,
    FeedbackAPIView,
    AnalyticsAPIView,
    UploadDocumentAPIView,
    DocumentListAPIView,
    DocumentDeleteAPIView,
    SwitchDomainAPIView,
)

urlpatterns = [
    path("ask/", AskAPIView.as_view(), name="ask"),
    path("feedback/", FeedbackAPIView.as_view(), name="feedback"),
    path("analytics/", AnalyticsAPIView.as_view(), name="analytics"),

    path("upload-document/", UploadDocumentAPIView.as_view(), name="upload_document"),
    path("my-documents/", DocumentListAPIView.as_view(), name="my_documents"),

    path("delete-document/<uuid:doc_id>/", DocumentDeleteAPIView.as_view(), name="delete_document"),
    path("switch-domain/", SwitchDomainAPIView.as_view(), name="switch_domain"),
]
