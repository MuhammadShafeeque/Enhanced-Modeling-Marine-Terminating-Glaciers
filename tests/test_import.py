import unittest


class TestPackageImport(unittest.TestCase):
    def test_import(self):
        """Test that the package can be imported."""
        try:
            import oggm_marine_terminating
            self.assertIsNotNone(oggm_marine_terminating.__name__)
            self.assertEqual(
                oggm_marine_terminating.__name__,
                "oggm_marine_terminating"
            )
        except ImportError:
            self.fail("Failed to import oggm_marine_terminating")
    
    def test_import_modules(self):
        """Test that individual modules can be imported."""
        try:
            from oggm_marine_terminating import flux_model
            # Check that the main model class is available
            self.assertTrue(hasattr(flux_model, "FluxBasedModelMarineFront"))
        except ImportError:
            self.fail("Failed to import oggm_marine_terminating modules")


if __name__ == '__main__':
    unittest.main()
