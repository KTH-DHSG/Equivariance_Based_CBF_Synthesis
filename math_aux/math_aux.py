"""

    Adrian Wiltz, 2025

"""

import numpy as np

def wrap_to_pi(angle):
    """
    Wraps an angle to the range [-pi, pi].

    Args:
        angle (float or array-like): The angle(s) to be wrapped.

    Returns:
        float or array-like: The wrapped angle(s) in the range [-pi, pi].
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi
