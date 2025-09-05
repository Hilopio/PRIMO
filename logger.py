from loguru import logger
import time

logger.remove()

logger.add(
    "petroscope.log",
    # format="<green>{time}</green> <level>{level}</level> <cyan>{message}</cyan>",
    rotation="10 MB",
    level="DEBUG"
)


def time_to_hms(time_in_seconds):
    hours, remainder = divmod(time_in_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours > 0:
        message = f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
    elif minutes > 0:
        message = f"{int(minutes)}m {seconds:.2f}s"
    else:
        message = f"{seconds:.2f}s"
    return message


def log_time(label, logger):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            message = time_to_hms(elapsed)
            logger.debug(f"{label} {message}")
            return result
        return wrapper
    return decorator
