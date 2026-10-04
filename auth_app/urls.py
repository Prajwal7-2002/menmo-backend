from django.urls import path
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView

from .views import ResetPasswordView, SignupView


class LoginView(TokenObtainPairView):
    throttle_scope = "auth"


urlpatterns = [
    path('signup', SignupView.as_view(), name='signup'),
    path('signup/', SignupView.as_view(), name='signup-slash'),

    path('login', LoginView.as_view(), name='login'),
    path('login/', LoginView.as_view(), name='login-slash'),

    path('refresh', TokenRefreshView.as_view(), name='token_refresh'),
    path('refresh/', TokenRefreshView.as_view(), name='token_refresh-slash'),

    path('reset-password', ResetPasswordView.as_view(), name='reset_password'),
    path('reset-password/', ResetPasswordView.as_view(), name='reset_password-slash'),
]
