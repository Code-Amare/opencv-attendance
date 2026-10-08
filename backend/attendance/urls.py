from django.urls import path
from .views import EvaluateAttendanceView

urlpatterns = [
    path("/", EvaluateAttendanceView.as_view()),
]
