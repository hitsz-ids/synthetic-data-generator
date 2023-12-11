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
    pass
