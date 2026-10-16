import logging

from django.conf import settings

logger = logging.getLogger(__name__)


class ProxyHopLogger:
    """
    Logs each distinct number of addresses seen in X-Forwarded-For (per worker).

    Per-IP rate limits need to know how many of those addresses were added by
    trusted proxies (Hugging Face's) rather than sent by the client. A normal
    browser sends none, so its requests show exactly the proxy count. Only the
    count and path are logged, never the addresses. Set TRUSTED_PROXIES to it.
    """

    def __init__(self, get_response):
        self.get_response = get_response
        self.seen = set()

    def __call__(self, request):
        xff = request.META.get("HTTP_X_FORWARDED_FOR", "")
        hops = len([a for a in xff.split(",") if a.strip()])
        if hops not in self.seen and len(self.seen) < 10:
            self.seen.add(hops)
            configured = settings.REST_FRAMEWORK.get("NUM_PROXIES")
            logger.info(
                "X-Forwarded-For carries %d address(es) (path %s); TRUSTED_PROXIES is %s",
                hops, request.path, "unset" if configured is None else configured,
            )
        return self.get_response(request)
