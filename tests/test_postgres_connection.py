import unittest

from model_service_common.postgres import postgres_hosts_and_ports


class PostgresHostsAndPortsTests(unittest.TestCase):
    def test_hosts_and_ports_are_aligned(self):
        self.assertEqual(
            postgres_hosts_and_ports("postgres-a,postgres-b", "5432,5433"),
            ("postgres-a,postgres-b", "5432,5433"),
        )

    def test_single_port_is_repeated(self):
        self.assertEqual(
            postgres_hosts_and_ports("postgres-a,postgres-b", "5432"),
            ("postgres-a,postgres-b", "5432,5432"),
        )


if __name__ == "__main__":
    unittest.main()
