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
        self.config = {
            "types": {
                "float": {
                    "allow_round": 2
                },
                "list": {
                    "check_length": True,
                    "length_diff_penalty": True
                }
            },
            "output": {
                "console": False,
                "file": False
            }
        }
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
        a = [1.23, 3, 'three', False, None, None]

        diff = self.compare.check(e, e)
        self.assertEqual(diff, NO_DIFF)

        diff = self.compare.check(e, a)
        self.assertEqual(
            {
                '_length': LengthsNotEqual(len(e), len(a), 2).explain(), # Warning! Length error weight is multiplied by the difference in lists lengths
                '_content': {
                    1: ValuesNotEqual(2, 3, 1).explain(),
                    3: ValuesNotEqual(True, False, 1).explain(),
                    'extra_4': ExtraListItem(None, None, 1).explain(),
                    'extra_5': ExtraListItem(None, None, 1).explain(),
                },
            },
            diff,
        )

    def test_list_length_difference(self):
        e = [1, 2, 3]
        a = [1, 2, 3, 4, 5]

        # Do not check lengths
        compare = Compare({
            "types": {
                "list": {
                    "check_length": False,
                }
            }
        })
        diff = compare.check(e, a)
        self.assertEqual({
            '_content': {
                'extra_3': ExtraListItem(None, 4, 1).explain(),
                'extra_4': ExtraListItem(None, 5, 1).explain(),
            },
        }, diff)

        # Check lengths but do not penalize
        compare = Compare({
            "types": {
                "list": {
                    "check_length": True,
                    "length_diff_penalty": False
                }
            }
        })
        diff = compare.check(e, a)
        self.assertEqual({
            '_length': LengthsNotEqual(len(e), len(a), 1).explain(),
            '_content': {
                'extra_3': ExtraListItem(None, 4, 1).explain(),
                'extra_4': ExtraListItem(None, 5, 1).explain(),
            },
        }, diff)

        # Check lengths and penalize by the difference in lengths of the lists (2)
        compare = Compare({
            "types": {
                "list": {
                    "check_length": True,
                    "length_diff_penalty": True
                }
            }
        })
        diff = compare.check(e, a)
        self.assertEqual({
            '_length': LengthsNotEqual(len(e), len(a), 2).explain(),
            '_content': {
                'extra_3': ExtraListItem(None, 4, 1).explain(),
                'extra_4': ExtraListItem(None, 5, 1).explain(),
            },
        }, diff)


    def test_deep_list_compare(self):
        # Test with nested lists
        # Items should be matched correctly using Hungarian algorithm
        e = [
            {'a': 1, 'b': 2},
            {'a': 2, 'b': 3},
            {'a': 3, 'b': 4},
        ]
        a = [
            {'a': 1, 'b': 2},
            {'a': 3, 'b': 4},
            {'a': 2, 'b': 3},
            {'a': 4, 'b': 5},
        ]

        result = self.compare.calculate_score(e, e)
        self.assertEqual(result.diff, NO_DIFF)

        result = self.compare.calculate_score(e, a)
        self.assertEqual(
            {
                '_length': LengthsNotEqual(len(e), len(a)).explain(),
                '_content': {
                    'extra_3': ExtraListItem(None, {'a': 4, 'b': 5}).explain(),
                },
            },
            result.diff
        )
        self.assertAlmostEqual(1 - (2 / 6), result.similarity)

        result = self.compare.calculate_score(a, e)
        self.assertEqual(
            {
                '_length': LengthsNotEqual(len(a), len(e)).explain(),
                '_content': {
                    3: MissingListItem({'a': 4, 'b': 5}, None).explain(),
                },
            },
            result.diff
        )
        self.assertAlmostEqual(1 - (2 / 8), result.similarity)

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

    def test_weights_basic(self):
        e = { 'int': 1, 'str': 'hi', 'list': [1, 2, 3], 'bool': True }
        a = { 'int': 1, 'str': 'guten tag', 'list': [2] }

        compare = Compare(self.config, weights={
            'int': 3,
            'str': 10,
            'list': {
                '_length': 0.3,
            },
        })

        result = compare.calculate_score(e, a)
        self.assertEqual(
            {
                'str': ValuesNotEqual('hi', 'guten tag', 10).explain(),
                'list': {
                    '_length': LengthsNotEqual(3, 1, 2 * 1 * 0.3).explain(), # Warning! Length error weight is multiplied by the difference in lists lengths and by _weight of the whole list!
                    '_content': {
                        0: MissingListItem(1, None, 1).explain(),
                        2: MissingListItem(3, None, 1).explain(),
                    },
                },
                'bool': KeyNotExist('bool', None, 1).explain(),
            },
            result.diff
        )
        self.assertEqual(13.6, result.failed_weighted)
        self.assertEqual(6, result.count)
        self.assertEqual(17, result.weighted_count)
        self.assertAlmostEqual(1 - (13.6 / 17), result.similarity)

    def test_weights_nested_objects(self):
        e = {
            'obj': {
                'nested_str': 'aloha',
                'nested_obj': { 'attr': 'Hi' }
            },
        }
        a = {
            'obj': {
                'nested_str': 'guten tag',
                'nested_obj': { 'attr': 'Hi2' }
            },
        }

        compare = Compare(self.config, weights={
            'obj': {
                '_weight': 4, # weight of the whole `obj` object
                'nested_str': 3,
                'nested_obj': {
                    '_weight': 2, # weight of the whole `nested_obj` object
                    'attr': 2,
                }
            }
        })

        result = compare.calculate_score(e, a)
        self.assertEqual(
            {
                'obj': {
                    'nested_str': ValuesNotEqual('aloha', 'guten tag', 4 * 3).explain(),
                    'nested_obj': {
                        'attr': ValuesNotEqual('Hi', 'Hi2', 4 * 2 * 2).explain(),
                    },
                },
            },
            result.diff
        )
        self.assertEqual(28, result.failed_weighted)
        self.assertEqual(2, result.count)
        self.assertEqual(12 + 16, result.weighted_count)
        self.assertAlmostEqual(1 - (28 / 28), result.similarity)

    def test_weights_different_syntax(self):
        e = {
            'obj': {'nested_str': 'aloha'},
            'list': [1, 2]
        }
        a = {
            'obj': {'nested_str': 'guten tag'},
            'list': [1, 4]
        }

        compare = Compare(self.config, weights={
            'obj': { '_weight': 3 },
            'list': { '_weight': 2 }
        })
        result = compare.calculate_score(e, a)
        self.assertEqual(5, result.failed_weighted)
        self.assertEqual(3, result.count)
        self.assertEqual(3 + 2 + 2, result.weighted_count)
        self.assertAlmostEqual(1 - (5 / 7), result.similarity)

        compare = Compare(self.config, weights={
            'obj': 3,
            'list': 2
        })
        result = compare.calculate_score(e, a)
        self.assertEqual(5, result.failed_weighted)
        self.assertEqual(3, result.count)
        self.assertEqual(3 + 2 + 2, result.weighted_count)
        self.assertAlmostEqual(1 - (5 / 7), result.similarity)

    def test_weights_lists_with_objects(self):
        e = {
            'list': [
                {'a': 1, 'b': 2},
                {'a': 2, 'b': 3},
            ]
        }
        a = {
            'list': [
                {'a': 1, 'b': 2},
                {'a': 999, 'b': 3},
                {'a': 4, 'b': 5},
            ]
        }

        compare = Compare(self.config, weights={
            'list': {
                '_weight': 4,
                '_length': 0.3,
                '_content': {
                    'a': 2
                }
            }
        })

        result = compare.calculate_score(e, a)
        self.assertEqual(
            {
                'list': {
                    '_length': LengthsNotEqual(2, 3, 4 * 1 * 0.3).explain(), # Warning! Length error weight is multiplied by the difference in lists lengths and by _weight of the whole list!
                    '_content': {
                        1: {
                            'a': ValuesNotEqual(2, 999, 4 * 2).explain(),
                        },
                        'extra_2': ExtraListItem(None, {'a': 4, 'b': 5},4).explain(),
                    },
                },
            },
            result.diff
        )
        self.assertEqual(13.2, result.failed_weighted)
        self.assertEqual(4, result.count)
        self.assertEqual(8 + 4 + 8 + 4, result.weighted_count)
        self.assertAlmostEqual(1 - (13.2 / 24), result.similarity)

    def test_weights_missing_and_extra(self):
        e = [{'a': 1}, {'a': 2}, {'a': 3}]
        a = [{'a': 2}, {'a': 5}]

        compare = Compare(self.config, weights={
            '_weight': 6,
            '_length': 5,
            '_missing': 4,
            '_extra': 3,
            '_content': {
                'a': 2,
            }
        })

        # Compare e and a
        result = compare.calculate_score(e, a)
        self.assertEqual(
            {
                '_length': LengthsNotEqual(3, 2, 6 * 5 * 1).explain(), # Warning! Length error weight is multiplied by the difference in lists lengths and by _weight of the whole list!
                '_content': {
                    0: {
                        'a': ValuesNotEqual(1, 5, 6 * 2).explain(),
                    },
                    2: MissingListItem({'a': 3}, None, 6 * 4).explain(),
                },
            },
            result.diff
        )
        self.assertEqual(66, result.failed_weighted)
        self.assertAlmostEqual(0, result.similarity) # The similarity is zero because the weighted number of errors is greater than the weighted count of attributes.

        # Compare a and e (vice versa)
        result = compare.calculate_score(a, e)
        self.assertEqual(
            {
                '_length': LengthsNotEqual(2, 3, 6 * 5 * 1).explain(), # Warning! Length error weight is multiplied by the difference in lists lengths and by _weight of the whole list!
                '_content': {
                    1: {
                        'a': ValuesNotEqual(5, 1, 6 * 2).explain(),
                    },
                    'extra_2': ExtraListItem(None, {'a': 3}, 6 * 3).explain(),
                },
            },
            result.diff
        )
        self.assertEqual(60, result.failed_weighted)
        self.assertAlmostEqual(0, result.similarity) # The similarity is zero because the weighted number of errors is greater than the weighted count of attributes.

    def test_weights_missing_and_extra_with_boost(self):
        # similar to prev
        e = [{'a': 1, 'b': 2}, {'a': 2, 'b': 3, 'c': 4, 'd': 5}]
        a = [{'a': 1, 'b': 2}]

        compare = Compare(self.config, weights={
            '_weight': 5,
            '_length': 4,
            '_missing': 3,
            '_boost_missing': True,
            '_extra': 2,
            '_boost_extra': True,
            '_content': {
                'c': 10
            }
        })

        # Compare e and a
        result = compare.calculate_score(e, a)
        self.assertEqual(
            {
                '_length': LengthsNotEqual(2, 1, 4 * 1 * 5).explain(), # Warning! Length error weight is multiplied by the difference in lists lengths and by _weight of the whole list!
                '_content': {
                    1: MissingListItem({'a': 2, 'b': 3, 'c': 4, 'd': 5}, None, 5 * 3 * 13).explain(), # _weight * _missing * _boost_missing (total weight of all attributes of the missing object)
                },
            },
            result.diff
        )
        self.assertEqual(4 * 1 * 5 + 5 * 3 * 13, result.failed_weighted)
        self.assertAlmostEqual(0, result.similarity) # the similarity is negative due to the boost, so it is 0 at the end

        # Compare a and e (vice versa)
        result = compare.calculate_score(a, e)
        self.assertEqual(
            {
                '_length': LengthsNotEqual(1, 2, 4 * 1 * 5).explain(), # Warning! Length error weight is multiplied by the difference in lists lengths and by _weight of the whole list!
                '_content': {
                    'extra_1': ExtraListItem(None, {'a': 2, 'b': 3, 'c': 4, 'd': 5}, 5 * 2 * 13).explain(), # _weight * _extra * _boost_missing (total weight of all attributes of the missing object)
                },
            },
            result.diff
        )
        self.assertEqual(4 * 1 * 5 + 5 * 2 * 13, result.failed_weighted)
        self.assertAlmostEqual(0, result.similarity) # the similarity is negative due to the boost, so it is 0 at the end


    def test_list_pairing_with_weights(self):
        e = [
            {'a': 1, 'b': 1, 'c': 1},
        ]
        a = [
            {'a': 2, 'b': 2, 'c': 1},
            {'a': 1, 'b': 1, 'c': 2},
            {'a': 2, 'b': 1, 'c': 1}, # closest to e[0]
        ]

        compare = Compare(self.config, weights={
            '_content': {
                'a': 1,
                'b': 1,
                'c': 10,
            }
        })

        # Should match e[0] with a[2] because c has the highest weight
        result = compare.calculate_score(e, a)
        self.assertEqual(
            {
                '_length': LengthsNotEqual(1, 3, 2).explain(), # Warning! Length error weight is multiplied by the difference in lists lengths and by _weight of the whole list!
                '_content': {
                    0: {
                        'a': ValuesNotEqual(1, 2, 1).explain(),
                    },
                    'extra_0': ExtraListItem(None, {'a': 2, 'b': 2, 'c': 1}, 1).explain(),
                    'extra_1': ExtraListItem(None, {'a': 1, 'b': 1, 'c': 2}, 1).explain(),
                },
            },
            result.diff
        )
        self.assertEqual(5, result.failed_weighted)
        self.assertAlmostEqual(1 + 1 + 10, result.weighted_count)
        self.assertAlmostEqual(1 - (5 / 12), result.similarity) # The similarity is zero because the weighted number of errors is greater than the weighted count of attributes.

    def test_list_pairing_with_weights_advanced(self):
        e = [
            { # should pair with a[1]
                'a': 1,
                'c': [
                    {'d': 4, 'e': 5},
                    {'d': 6, 'e': 7},
                ]
            },
        ]
        a = [
            { # this does not pair with e[0] because there are two different attributes in `c`, which matters more than matching `a` attribute due to weights
                'a': 1,
                'c': [
                    {'d': 999, 'e': 5},
                    {'d': 999, 'e': 7},
                ]
            },
            {
                'a': 999, # this is different, but it does not matter because it has low weight
                'c': [ # list items order is different, but it does not matter at all, pairing should correctly map (d: 4, e: 5) with (d: 999, e: 5)
                    {'d': 6, 'e': 7},
                    {'d': 999, 'e': 5}, # there is a difference in one attribute, but it should have small effect on pairings
                ]
            }
        ]

        compare = Compare(self.config, weights={
            '_content': {
                'a': 1,
                'c': {
                    '_weight': 10
                }
            }
        })

        # Should match e[0] with a[1] because:
        # - `a` has low weight and does not matter as much as `c`
        # - `c` items are in different order, but it should have zero effect on pairing
        # - `c` has one different attribute, but it should have smaller effect than two different attributes in a[0]
        result = compare.calculate_score(e, a)
        self.assertEqual(
            {
                '_length': LengthsNotEqual(1, 2, 1).explain(), # Warning! Length error weight is multiplied by the difference in lists lengths and by _weight of the whole list!
                '_content': {
                    0: {
                        'a': ValuesNotEqual(1, 999, 1).explain(),
                        'c': {
                            '_content': {
                                0: {'d': ValuesNotEqual(4, 999, 10).explain()},
                            }
                        },
                    },
                    'extra_0': ExtraListItem(None, {'a': 1, 'c': [{'d': 999, 'e': 5}, {'d': 999, 'e': 7}]}, 1).explain(),
                }
            },
            result.diff
        )
        self.assertEqual(13, result.failed_weighted)
        self.assertAlmostEqual(1 + 4 * 10, result.weighted_count)
        self.assertAlmostEqual(1 - (13 / 41), result.similarity) # The similarity is zero because the weighted number of errors is greater than the weighted count of attributes.


    def test_list_pairing_threshold(self):
        # e[0] is similar to a[1] with similarity 0.5
        # e[1] is not similar to a[0], similarity is 0.0
        # e[2] is same as a[2] with similarity 1.0
        e = [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}, {'a': 5, 'b': 6}]
        a = [{'a': 999, 'b': 999}, {'a': 1, 'b': 999}, {'a': 5, 'b': 6}]

        # All should be paired because similarities of both pairs are above the threshold
        compare = Compare(self.config, weights={'_pairing_threshold': 0})
        result = compare.calculate_score(e, a)
        self.assertEqual({'_content': {
            0: {
                'b': ValuesNotEqual(2, 999, 1).explain(),
            },
            1: {
                'a': ValuesNotEqual(3, 999, 1).explain(),
                'b': ValuesNotEqual(4, 999, 1).explain(),
            }
        }}, result.diff)
        self.assertEqual(3, result.failed_weighted)

        # Items e[0] and e[2] are paired, but e[1] is not paired due to the threshold set to 0.4
        # Item e[1] should not be paired and throws MissingListItem and ExtraListItem instead of ValuesNotEqual
        compare = Compare(self.config, weights={'_pairing_threshold': 0.4})
        result = compare.calculate_score(e, a)
        self.assertEqual({'_content': {
            0: {
                'b': ValuesNotEqual(2, 999, 1).explain(), # e[0] is paired with a[1], but the match is not perfect
            },
            1: MissingListItem({'a': 3, 'b': 4}, None, 1).explain(), # MissingListItem for e[1] instead of ValuesNotEqual
            'extra_0': ExtraListItem(None, {'a': 999, 'b': 999}, 1).explain(), # ExtraListItem for a[0] instead of ValuesNotEqual
        }}, result.diff)

        # Only e[2] is paired because the threshold requires perfect match
        # Other items should not be paired and throws MissingListItem and ExtraListItem instead of ValuesNotEqual
        compare = Compare(self.config, weights={'_pairing_threshold': 1.0})
        result = compare.calculate_score(e, a)
        self.assertEqual({'_content': {
            0: MissingListItem({'a': 1, 'b': 2}, None, 1).explain(), # MissingListItem for e[0] instead of ValuesNotEqual
            1: MissingListItem({'a': 3, 'b': 4}, None, 1).explain(), # MissingListItem for e[1] instead of ValuesNotEqual
            'extra_0': ExtraListItem(None, {'a': 999, 'b': 999}, 1).explain(), # ExtraListItem for a[0] instead of ValuesNotEqual
            'extra_1': ExtraListItem(None, {'a': 1, 'b': 999}, 1).explain(), # ExtraListItem for a[1] instead of ValuesNotEqual
        }}, result.diff)



if __name__ == '__main__':
    unittest.main()
