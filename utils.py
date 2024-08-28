import time
import openai
import anthropic
import google.api_core.exceptions as google_exceptions

# define a retry decorator
def retry_with_linear_backoff(
    delay: float = 90,
    max_retries: int = 10,
    errors: tuple = (
        openai.RateLimitError,
        anthropic.RateLimitError,
        google_exceptions.ResourceExhausted,
    ),
):
    """Retry a function with linear backoff."""
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Initialize variables
            num_retries = 0

            # Loop until a successful response or max_retries is hit or an exception is raised
            while True:
                try:
                    return func(*args, **kwargs)

                # Retry on specified errors
                except errors as e:
                    # Increment retries
                    num_retries += 1

                    # Check if max retries has been reached
                    if num_retries > max_retries:
                        raise Exception(
                            f"Maximum number of retries ({max_retries}) exceeded."
                        )

                    # Sleep for the delay
                    time.sleep(delay)

                # Raise exceptions for any errors not specified
                except Exception as e:
                    raise e

        return wrapper
    return decorator
