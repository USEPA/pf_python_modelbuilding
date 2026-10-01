from unittest import TestCase

from util.serialization_compat import deserialize_model


class SerializationCompatTests(TestCase):
    def test_deserialize_model_loads_numpy_2_numeric_global(self):
        payload = b"cnumpy._core.numeric\n_frombuffer\n."

        restored = deserialize_model(payload)

        self.assertEqual(restored.__name__, "_frombuffer")

    def test_deserialize_model_loads_numpy_2_multiarray_global(self):
        payload = b"cnumpy._core.multiarray\n_reconstruct\n."

        restored = deserialize_model(payload)

        self.assertEqual(restored.__name__, "_reconstruct")
