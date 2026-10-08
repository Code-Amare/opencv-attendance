from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny
from django.contrib.auth import get_user_model

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
