import os
from redis import Redis
from rq import Queue


DEFAULT_QUEUE_NAME = os.environ.get("RQ_QUEUE", "scriptgen").strip() or "scriptgen"


def get_redis() -> Redis:
    """
    Render: use REDIS_URL from a Redis add-on / external Redis.
    Local dev: defaults to redis://localhost:6379/0.
    """
    redis_url = (os.environ.get("REDIS_URL") or "redis://localhost:6379/0").strip()
    return Redis.from_url(redis_url)


def get_queue() -> Queue:
    return Queue(name=DEFAULT_QUEUE_NAME, connection=get_redis(), default_timeout=60 * 60 * 6)

