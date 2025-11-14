"""
Test runner for CulturalGaN project.
Runs all unit tests and generates a report.
"""

import sys
import unittest
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_all_tests(verbosity=2):
    """
    Discover and run all tests.

    Args:
        verbosity: Verbosity level (0-2)

    Returns:
        Test result object
    """
    # Discover tests
    loader = unittest.TestLoader()
    start_dir = Path(__file__).parent
    suite = loader.discover(start_dir, pattern='test_*.py')

    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)

    return result


def main():
    """Main entry point."""
    print("=" * 70)
    print("CulturalGaN Test Suite")
    print("=" * 70)
    print()

    result = run_all_tests(verbosity=2)

    print()
    print("=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print()

    if result.wasSuccessful():
        print("✓ All tests passed!")
        return 0
    else:
        print("❌ Some tests failed.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
