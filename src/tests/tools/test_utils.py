import logging
import unittest

from src.tools import utils


class TestUtils(unittest.TestCase):

    # Function ensure_input_list
    def test_ensure_input_list_single_value(self):
        obj = 1
        expected = [1]
        result = utils.ensure_input_list(obj)
        assert expected == result

    def test_ensure_input_list_already_list(self):
        obj = [1]
        expected = [1]
        result = utils.ensure_input_list(obj)
        self.assertEqual(expected, result)

    def test_ensure_input_list_none(self):
        obj = None
        expected = []
        result = utils.ensure_input_list(obj)
        self.assertEqual(expected, result)

    # Function perform_dict_union_recursively
    def test_perform_dict_union_recursively_correct(self):
        dict_x = {'a': {'x': 8, 'y': [2, 3]}, 'b': 0}
        dict_y = {'a': {'z': [3, 4]}, 'c': 5}
        expected = {'a': {'x': 8, 'y': [2, 3], 'z': [3, 4]}, 'b': 0, 'c': 5}
        result = utils.perform_dict_union_recursively(dict_x, dict_y)
        self.assertEqual(expected, result)

    def test_perform_dict_union_recursively_error_single(self):
        dict_x = {'a': 1}
        dict_y = {'a': 2}
        expected_str = (
            'Pure union cannot be made due to key ["a"]. '
            'Please review both dictionaries.')
        with self.assertRaises(AttributeError) as context:
            utils.perform_dict_union_recursively(dict_x, dict_y)
        self.assertEqual(expected_str, str(context.exception))

    def test_perform_dict_union_recursively_error_dict(self):
        dict_x = {'a': {'x': 1}}
        dict_y = {'a': 2}
        expected_str = (
            'Pure union cannot be made due to key ["a"]. '
            'Please review both dictionaries.')
        with self.assertRaises(AttributeError) as context:
            utils.perform_dict_union_recursively(dict_x, dict_y)
        self.assertEqual(expected_str, str(context.exception))

    # Function import_library
    def test_import_library_without_params(self):
        module = 'logging.Formatter'
        result = utils.import_library(module)
        self.assertIsInstance(result, logging.Formatter)

    def test_import_library_with_empty_params(self):
        module = 'logging.Formatter'
        params = None
        result = utils.import_library(module, params)
        self.assertIsInstance(result, logging.Formatter)

    def test_import_library_with_params(self):
        module = 'logging.Formatter'
        params = {'fmt': '%(asctime)s - %(message)s'}
        result = utils.import_library(module, params)
        self.assertIsInstance(result, logging.Formatter)


if __name__ == '__main__':
    unittest.main()
