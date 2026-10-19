from django.urls import path

from .views import ReviewDocumentView

urlpatterns = [path("reviews/", ReviewDocumentView.as_view(), name="review-document")]
