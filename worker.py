
import os
import redis
from rq import Worker, Queue, Connection
from app import app
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - WORKER - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import time

listen = ['default']
redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')

# Retry connection logic for production resilience
conn = None
for i in range(10):
    try:
        conn = redis.from_url(redis_url)
        conn.ping()
        logger.info(f"Connected to Redis at {redis_url}")
        break
    except redis.exceptions.ConnectionError:
        logger.warning(f"Waiting for Redis to become available... ({i + 1}/10)")
        time.sleep(2)

if not conn:
    logger.error("Could not connect to Redis. Worker exiting.")
    exit(1)

if __name__ == '__main__':
    logger.info(f"Starting RQ Worker listening on {listen}...")
    with Connection(conn):
        worker = Worker(list(map(Queue, listen)))
        worker.work()
