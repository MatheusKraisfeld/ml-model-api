import unittest
from iris_model_api import validate_input

class TestValidateInpu(unittest.TestCase):
    def test_validate_input(self):
        self.assertEqual(validate_input(5.1, 3.5, 1.4, 0.2), True)
        self.assertEqual(validate_input(5.1, 1.5, 1.4, 0.2), False)
        self.assertEqual(validate_input(0, 0, 0, 0), False)
        self.assertEqual(validate_input(1.1, 1.1, 1.1, 1.1), False)

if __name__ == '__main__':
    unittest.main()