import unittest
import json

from jsoncomparison import (
    NO_DIFF,
    Compare,
    KeyNotExist,
    LengthsNotEqual,
    TypesNotEqual,
    ValueNotFound,
    ValuesNotEqual,
)
from jsoncomparison.errors import UnexpectedKey, MissingListItem, ExtraListItem

from . import load_json


class CompareTestCase(unittest.TestCase):

    def setUp(self):
        self.config = load_json('compare/config.json')
        self.compare = Compare(self.config)

    def test_compare_int(self):
        diff = self.compare.check(1, 1)
        self.assertEqual(NO_DIFF, diff)

        diff = self.compare.check(1, 2)
        self.assertEqual(ValuesNotEqual(1, 2).explain(), diff)

    def test_compare_str(self):
        diff = self.compare.check('str', 'str')
        self.assertEqual(NO_DIFF, diff)

        diff = self.compare.check('str1', 'str2')
        self.assertEqual(ValuesNotEqual('str1', 'str2').explain(), diff)

    def test_compare_float(self):
        diff = self.compare.check(1.2, 1.2)
        self.assertEqual(NO_DIFF, diff)

        diff = self.compare.check(1.23456, 1.23)
        self.assertEqual(NO_DIFF, diff)

        diff = self.compare.check(1.2, 1.3)
        self.assertEqual(ValuesNotEqual(1.2, 1.3).explain(), diff)

    def test_compare_bool(self):
        diff = self.compare.check(True, True)
        self.assertEqual(NO_DIFF, diff)

        diff = self.compare.check(True, False)
        self.assertEqual(ValuesNotEqual(True, False).explain(), diff)

    def test_compare_dict_diff(self):
        e = {'int': 1, 'str': 'Hi', 'float': 1.23, 'bool': True}
        a = {'int': 2, 'str': 'Hi', 'float': 1}

        diff = self.compare.check(e, e)
        self.assertEqual(NO_DIFF, diff)

        diff = self.compare.check(e, a)
        self.assertEqual(
            {
                'int': ValuesNotEqual(1, 2).explain(),
                'float': TypesNotEqual(1.23, 1).explain(),
                'bool': KeyNotExist('bool', None).explain(),
            },
            diff,
        )

    def test_compare_dict_diff_unexpected(self):
        e = {'int': 2, 'str': 'Hi', 'float': 1}
        a = {'int': 1, 'str': 'Hi', 'float': 1.23, 'bool': True}

        diff = self.compare.check(e, a)
        self.assertEqual(
            {
                'int': ValuesNotEqual(2, 1).explain(),
                'float': TypesNotEqual(1, 1.23).explain(),
                'bool': UnexpectedKey(None, 'bool').explain(),
            },
            diff,
        )

    def test_list_compare(self):
        e = [1.23, 2, 'three', True]
        a = [1.23, 3, 'three', False, None]

        diff = self.compare.check(e, e)
        self.assertEqual(diff, NO_DIFF)

        diff = self.compare.check(e, a)
        self.assertEqual(
            {
                '_length': LengthsNotEqual(len(e), len(a)).explain(),
                '_content': {
                    1: ValuesNotEqual(2, 3, 1).explain(),
                    3: ValuesNotEqual(True, False, 1).explain(),
                    4: ExtraListItem(None, None, 1).explain(),
                },
            },
            diff,
        )

    def test_depp_list_compare(self):
        # Test with nested lists
        # Items should be matched correctly using Hungarian algorithm

        e = [
            {'key': 1, 'value': 2},
            {'key': 2, 'value': 3},
            {'key': 3, 'value': 4},
        ]
        a = [
            {'key': 1, 'value': 2},
            {'key': 3, 'value': 4},
            {'key': 2, 'value': 3},
            {'key': 4, 'value': 5},
        ]

        diff = self.compare.check(e, e)
        self.assertEqual(diff, NO_DIFF)

        diff = self.compare.check(e, a)
        self.assertEqual(
            {
                '_length': LengthsNotEqual(len(e), len(a)).explain(),
                '_content': {
                    3: ExtraListItem(None, {'key': 4, 'value': 5}).explain(),
                },
            },
            diff
        )

        diff = self.compare.check(a, e)
        self.assertEqual(
            {
                '_length': LengthsNotEqual(len(a), len(e)).explain(),
                '_content': {
                    3: MissingListItem({'key': 4, 'value': 5}, None).explain(),
                },
            },
            diff
        )

    def test_prepare_method(self):
        e = [1, 2, 3, 4]
        p = self.compare.prepare(e)

        self.assertTrue(e == p)
        self.assertTrue(e is not p)

    def test_compare_deep_data(self):
        rules = load_json('compare/rules.json')
        actual = load_json('compare/actual.json')
        expected = load_json('compare/expected.json')

        result = Compare(self.config, rules).calculate_score(expected, actual)

        # print('')
        # print(f'failed {result.failed} of {result.count}')
        # print('similarity', result.similarity)
        # print('diff', result.diff)

        self.assertEqual(NO_DIFF, result.diff)
        self.assertEqual(0, result.failed)
        self.assertEqual(1.0, result.similarity)

    def test_simple_list_with_dicts(self):
        expected = [
            {'a': "xxx"},
            {'a': "yyy"},
        ]
        actual = [
            {'a': "zzz"},
            {'a': "yyy"},
        ]
        diff = Compare().check(expected, actual)
        self.assertEqual(
            {
                '_content': {
                    0: {'a': ValuesNotEqual('xxx', 'zzz').explain()},
                },
            },
            diff,
        )

    # The next two tests represent some current behaviour that probably
    # is incorrect so they are left here as documentation for future changes
    # See https://github.com/rugleb/JsonCompare/pull/37#issuecomment-1821786007

    def test_list_with_dicts(self):
        expected = [
            {'a': "xxx"},
            {'a': "iii"},
            {'a': "yyy"},
            {'a': "jjj"},
        ]
        actual = [
            {'a': "zzz"},
            {'a': "yyy"},
            {'a': "xxx"},
            {'a': "eee"},
        ]
        diff = Compare().check(expected, actual)
        self.assertEqual(
            {
                '_content': {
                    1: {'a': ValuesNotEqual('iii', 'zzz').explain()},
                    3: {'a': ValuesNotEqual('jjj', 'eee').explain()},
                },
            },
            diff,
        )

    def test_list_with_dicts_with_duplicates(self):
        expected = [
            {'a': "xxx"}, # -> 3
            {'a': "iii"}, # -> 2
            {'a': "xxx"},
            {'a': "jjj"},
        ]
        actual = [
            {'a': "zzz"},
            {'a': "iii"},
            {'a': "xxx"},
            {'a': "iii"},
        ]
        diff = Compare().check(expected, actual)
        self.assertEqual(
       {
                '_content': {
                    2: {'a': ValuesNotEqual('xxx', 'zzz').explain()},
                    3: {'a': ValuesNotEqual('jjj', 'iii').explain()},
                },
            },
            diff
        )

    def test_weights(self):
        e = {
            'int': 1,
            'str': {
                'not_nested': 'aloha',
                'nested': { 'attr': 'Hi' }
            },
            'list': [
                {'a': 1, 'b': 2},
                {'a': 3, 'b': 4},
            ],
            'bool': True
        }
        a = {
            'int': 2,
            'str': {
                'not_nested': 'guten tag',
                'nested': { 'attr': 'Hi2' }
            },
            'list': [
                {'a': 1, 'b': 2},
            ]
        }

        compare = Compare(self.config, weights={
            'int': 3,
            'str': {
                '_weight': 10,
                'nested': {
                    'attr': 2,
                }
            },
            'list': {
                '_length': 0.3,
                '_list': {
                    'a': 5
                }
            },
        })

        result = compare.calculate_score(e, e)
        self.assertEqual(result.diff, NO_DIFF)

        result = compare.calculate_score(e, a)
        self.assertEqual(
            {
                'int': ValuesNotEqual(1, 2, 3).explain(),
                'str': {
                    'not_nested': ValuesNotEqual('aloha', 'guten tag', 10).explain(),
                    'nested': {
                        'attr': ValuesNotEqual('Hi', 'Hi2', 20).explain(),
                    },
                },
                'list': {
                    '_length': LengthsNotEqual(2, 1, 1 * 0.3).explain(),
                    '_content': {
                        1: MissingListItem({'a': 3, 'b': 4}, None).explain(),
                    },
                },
                'bool': KeyNotExist('bool', None).explain(),
            },
            result.diff
        )
        self.assertEqual(35.3, result.failed_weighted)


if __name__ == '__main__':
    unittest.main()
