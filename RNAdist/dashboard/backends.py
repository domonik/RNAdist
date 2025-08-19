from dash import DiskcacheManager, CeleryManager
from RNAdist.dashboard import CONFIG
import os



if CONFIG["REDIS"]["URL"]:
    URL = CONFIG["REDIS"]["URL"]
    try:
        from celery import Celery
    except ImportError:
        raise ImportError("The optional dependency celery is not installed. Install it via pip install celery[redis]")
    print(f"Managing background tasks via celery and URL - {URL}")
    celery = Celery(__name__, broker=URL, backend=URL)
    BACKGROUND_CALLBACK_MANAGER = CeleryManager(celery)

else:
    # Diskcache for non-production apps when developing locally
    import diskcache
    celery = None
    cache = diskcache.Cache(os.path.join(CONFIG["backend"]["name"], "cache"))
    BACKGROUND_CALLBACK_MANAGER = DiskcacheManager(cache)
