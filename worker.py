"""
RQ worker entrypoint for Render.

Start command example:
  python worker.py
or:
  rq worker scriptgen
"""

import os
from rq import Worker

from jobs import get_queue, get_redis, DEFAULT_QUEUE_NAME


def main() -> None:
    # Allow overriding queue names via env, but default to one queue.
    queues = [os.environ.get("RQ_QUEUE", DEFAULT_QUEUE_NAME)]
    w = Worker(queues, connection=get_redis())
    w.work(with_scheduler=False)


if __name__ == "__main__":
    main()

