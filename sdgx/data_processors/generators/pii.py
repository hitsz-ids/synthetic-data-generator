"""
The PII Generator module is designed to **generate** columns of type PII.
Random generation is too simple to be reasonable in many use cases.
This module is mainly responsible for:
- providing batch generation methods for different types of PII objects (columns).
- Providing randomised generation methods for different types of PII objects.
- generating PII objects with constraints such as geography, attribution, etc. as inputs
"""

from sdgx.data_processors.generators.base import Generator


class PIIGenerator(Generator):
    """
    The PIIGenerator class is a subclass of the Generator class. It is designed to generate PII (Personally Identifiable Information) objects.

    This class is responsible for:
        - providing batch generation methods for different types of PII objects (columns).
        - providing randomised generation methods for different types of PII objects.
        - generating PII columns with constraints such as geography, attribution, etc.
    """
