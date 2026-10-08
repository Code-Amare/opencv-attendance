from django.urls import path
from .views import EvaluateAttendanceView, CreateSessionView

urlpatterns = [
    path("", EvaluateAttendanceView.as_view()),
    path("session/create/", CreateSessionView.as_view()),
]
