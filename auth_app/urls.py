from django.urls import path
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView
from .views import SignupView, ResetPasswordView

urlpatterns = [
    path('signup', SignupView.as_view(), name='signup'),
    path('signup/', SignupView.as_view(), name='signup-slash'),

    path('login', TokenObtainPairView.as_view(), name='login'),
    path('login/', TokenObtainPairView.as_view(), name='login-slash'),

    path('refresh', TokenRefreshView.as_view(), name='token_refresh'),
    path('refresh/', TokenRefreshView.as_view(), name='token_refresh-slash'),

    path('reset-password', ResetPasswordView.as_view(), name='reset_password'),
    path('reset-password/', ResetPasswordView.as_view(), name='reset_password-slash'),
]
