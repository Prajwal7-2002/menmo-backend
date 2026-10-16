from rest_framework.throttling import AnonRateThrottle, UserRateThrottle


class AskDailyThrottle(UserRateThrottle):
    """Daily cap on LLM-backed messages per user (protects LLM spend)."""

    scope = "ask_daily"


class SignupDailyThrottle(AnonRateThrottle):
    """Daily cap on new accounts per client, so the per-user caps can't be dodged with fresh accounts."""

    scope = "signup"
