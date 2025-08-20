import unittest


class TestPackageImport(unittest.TestCase):
    def test_import(self):
        """Test that the package can be imported."""
        try:
            import oggm_marine_terminating
            self.assertIsNotNone(oggm_marine_terminating.__name__)
        except ImportError:
            self.fail("Failed to import oggm_marine_terminating")


if __name__ == '__main__':
    unittest.main()
