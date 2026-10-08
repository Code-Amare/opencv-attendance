from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny
from django.contrib.auth import get_user_model
from django.utils import timezone

from .models import Attendance, AttendanceSession
from .serializers import AttendanceSerializer

User = get_user_model()


class EvaluateAttendanceView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        user_id = request.data.get("user_id", "")
        session_id = request.data.get("session_id", "")

        if not user_id or not session_id:
            return Response(
                {"error": "user_id and session_id are required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        user = User.objects.filter(id=user_id).first()
        if not user:
            return Response(
                {"error": "Invalid user_id"}, status=status.HTTP_400_BAD_REQUEST
            )

        session = AttendanceSession.objects.filter(id=session_id).first()
        if not session:
            return Response(
                {"error": "Invalid session_id"}, status=status.HTTP_400_BAD_REQUEST
            )

        now = timezone.now()
        if now <= session.ended_at:
            attendance_status = Attendance.Status.PRESENT
        elif now <= session.ended_at + session.late_time:
            attendance_status = Attendance.Status.LATE
        else:
            attendance_status = Attendance.Status.ABSENT

        created, attendance = Attendance.objects.get_or_create(
            user=user,
            session=session,
            defaults={
                "status": attendance_status,
            },
        )

        if not created:
            return Response(
                {
                    "attendance": AttendanceSerializer(attendance).data,
                    "is_attendance_taken_before": True,
                },
                status=status.HTTP_200_OK,
            )

        return Response(
            {
                "attendance": AttendanceSerializer(attendance).data,
                "is_attendance_taken_before": False,
            },
            status=status.HTTP_201_CREATED,
        )
