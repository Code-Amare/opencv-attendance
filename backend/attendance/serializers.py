from django.utils import timezone
from rest_framework import serializers

from .models import Attendance, AttendanceSession


class AttendanceSerializer(serializers.ModelSerializer):
    user_name = serializers.CharField(
        source="user.first_name",
        read_only=True,
    )

    session_name = serializers.CharField(
        source="session.name",
        read_only=True,
    )

    class Meta:
        model = Attendance
        fields = [
            "id",
            "session",
            "session_name",
            "user",
            "user_name",
            "status",
            "recognized_at",
        ]
        read_only_fields = [
            "id",
            "status",
            "recognized_at",
        ]

    def validate(self, attrs):
        session = attrs["session"]

        if session.status != session.Status.ACTIVE:
            raise serializers.ValidationError("Attendance session is not active.")

        now = timezone.now()

        if session.ended_at and now > session.ended_at + session.late_time:
            raise serializers.ValidationError(
                "Attendance session has ended and the late period has expired."
            )

        return attrs

    def create(self, validated_data):
        session = validated_data["session"]
        now = timezone.now()

        if session.ended_at and now > session.ended_at:
            status = Attendance.Status.LATE
        else:
            status = Attendance.Status.PRESENT

        return Attendance.objects.create(
            **validated_data,
            status=status,
        )


class AttendanceSessionSerializer(serializers.ModelSerializer):
    class Meta:
        model = AttendanceSession
        fields = [
            "id",
            "created_by",
            "name",
            "started_at",
            "ended_at",
            "late_time",
            "status",
            "created_at",
        ]
        read_only_fields = [
            "id",
            "created_by",
            "status",
            "created_at",
        ]

    def validate(self, attrs):
        started_at = attrs.get("started_at")
        ended_at = attrs.get("ended_at")

        if started_at and ended_at and ended_at <= started_at:
            raise serializers.ValidationError(
                {"ended_at": "ended_at must be later than started_at."}
            )

        return attrs
