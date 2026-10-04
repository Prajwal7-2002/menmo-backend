from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError
from rest_framework import generics
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response

from .serializers import SignupSerializer


class SignupView(generics.CreateAPIView):
    serializer_class = SignupSerializer
    permission_classes = [AllowAny]
    throttle_scope = "auth"

    def post(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return Response({"detail": "User created successfully"}, status=201)


class ResetPasswordView(generics.GenericAPIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        old_password = request.data.get("old_password")
        new_password = request.data.get("new_password") or ""
        user = request.user

        if not user.check_password(old_password):
            return Response({"detail": "Old password is incorrect"}, status=400)
        try:
            validate_password(new_password, user=user)
        except ValidationError as e:
            return Response({"detail": " ".join(e.messages)}, status=400)

        user.set_password(new_password)
        user.save()
        return Response({"detail": "Password updated successfully"}, status=200)
