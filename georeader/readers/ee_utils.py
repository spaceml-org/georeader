try:
    import ee
except ImportError:
    raise ImportError("Please install the package 'earthengine-api' to use this module: pip install earthengine-api")

import threading

class EETimeoutError(Exception):
    """Exception raised when an Earth Engine getInfo() call times out."""
    pass
from typing import Callable, Any

DEFAULT_EE_TIMEOUT = 120  # seconds

def gee_method_with_timeout(method:Callable[[], Any], 
                            timeout: float = DEFAULT_EE_TIMEOUT) -> Any:
    """
    Wrapper around calling the Earth Engine API.

    Args:
        method (Callable[[Any], Any]): The Earth Engine method to call, e.g., `lambda _: obj.getInfo()`
        timeout (float): Maximum time to wait for calling the method. 
            Defaults to 120 seconds.

    Returns:
        dict: The result of the call.
        
    Raises:
        EETimeoutError: If the  call exceeds the timeout duration.
    """
    result = [None]
    exception = [None]
    
    def target():
        try:
            result[0] = method()
        except Exception as e:
            exception[0] = e
    
    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout)
    
    if thread.is_alive():
        # Thread is still running, timeout occurred
        raise EETimeoutError(
            f"Earth Engine getInfo() call exceeded timeout of {timeout} seconds. "
            "The operation may still be running on the server."
        )
    
    if exception[0] is not None:
        raise exception[0]
    
    return result[0]
