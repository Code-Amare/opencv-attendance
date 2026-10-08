from django.urls import path
from .views import EvaluateAttendanceView, CreateSessionView, AttendanceListView

urlpatterns = [
    path("", EvaluateAttendanceView.as_view()),
    path("session/create/", CreateSessionView.as_view()),
    path("sessions/", AttendanceListView.as_view()),
]
