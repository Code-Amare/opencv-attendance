from rest_framework import serializers
from .models import User


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = [
            "id",
            "username",
            "email",
            "first_name",
            "last_name",
            "section",
            "profile_picture",
            "phone_number",
            "date_of_birth",
            "email_verified",
            "two_factor_enabled",
        ]
        read_only_fields = [
            "id",
            "email_verified",
            "two_factor_enabled",
        ]