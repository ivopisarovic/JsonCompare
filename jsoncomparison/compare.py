import copy
import json
from typing import Optional
import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import Config
from .errors import (
    KeyNotExist,
    LengthsNotEqual,
    TypesNotEqual,
    UnexpectedKey,
    ValuesNotEqual, MissingListItem, ExtraListItem,
)
from .ignore import Ignore
from .utils import get_boolean, is_suppressed

NO_DIFF: dict = {}
NO_RULES: dict = {}
NO_WEIGHTS: dict = {}

DEFAULT_CONFIG = {
    'output': {
        'console': False,
        'file': {
            'allow_nan': True,
            'ensure_ascii': True,
            'indent': 4,
            'name': None,
            'skipkeys': True,
        },
    },
    'types': {
        'float': {
            'allow_round': 2,
        },
        'list': {
            'check_length': True,
        },
    },
}

class Result:

    __slots__ = ("_failed", "_weighted_failed", "_count", "_weighted_count", "_diff")

    def __init__(self, diff, count, weighted_count, failed, weighted_failed):
        self._diff = diff
        self._count = count
        self._weighted_count = weighted_count
        self._failed = failed
        self._weighted_failed = weighted_failed

    @property
    def failed(self):
        return self._failed

    @property
    def weighted_failed(self):
        return self._weighted_failed

    @property
    def count(self):
        return self._count

    @property
    def weighted_count(self):
        return self._weighted_count

    @property
    def similarity(self):
        if self._weighted_count == 0:
            return 0

        similarity = (self._weighted_count - self._weighted_failed) / self._weighted_count

        # sometimes, similarity can be negative, so we need to return 0 in this case
        # for example, when there are extra items in the second list
        return max(0, similarity)

    @property
    def diff(self):
        return self._diff


class Compare:

    __slots__ = ("_config", "_rules", "_weights")

    def __init__(
        self,
        config: Optional[dict] = None,
        rules: Optional[dict] = None,
        weights: Optional[dict] = None,
    ):
        if not config:
            config = DEFAULT_CONFIG
        if not rules:
            rules = NO_RULES
        if not weights:
            weights = NO_WEIGHTS

        self._config = Config(config)
        self._rules = rules
        self._weights = weights

    def check(self, expected, actual):
        e = self.prepare(expected)
        a = self.prepare(actual)
        weight = self._get_root_weight()
        suppressed = is_suppressed(self._weights)
        diff = self._diff(e, a, weight, self._weights, suppressed)
        self.report(diff)
        return diff

    def calculate_score(self, expected, actual):
        diff = self.check(expected, actual)
        weight = self._get_root_weight()
        return self._create_result(diff, expected, weight, self._weights)

    def _create_result(self, diff, expected, weight, weights):
        filtered_diff = self._without_suppressed_errors(diff)
        count = self._attributes_count(expected)
        weighted_count = self._weighted_attributes_count(expected, weight, weights)
        failed = self._count_failed(diff, False)
        weighted_failed = self._count_failed(diff, True)
        return Result(filtered_diff, count, weighted_count, failed, weighted_failed)

    def _without_suppressed_errors(self, diff):
        # Remove errors that are suppressed by setting `_suppress` in weights

        if is_suppressed(diff): # When the whole dict is suppressed, return {}
            return None

        if isinstance(diff, dict):
            result = {}
            for k, v in diff.items():
                if not is_suppressed(v):
                    result[k] = self._without_suppressed_errors(v)
            return self._without_empties(result)

        return diff

    def _count_failed(self, d, weighted):
        if isinstance(d, dict) and '_error' in d:
            if weighted and '_weight' in d:
                return d['_weight']
            else:
                return 1
        else:
            return sum(self._count_failed(v, weighted) for v in d.values())

    def _get_root_weight(self):
        return self._weights['_weight'] if '_weight' in self._weights else 1

    @staticmethod
    def _get_weight(weights, key):
        if isinstance(weights, (int, float)):
            return weights

        if key in weights:
            if isinstance(weights[key], dict):
                return weights[key]['_weight'] if '_weight' in weights[key] else 1
            elif isinstance(weights[key], (int, float)):
                return weights[key]
            else:
                raise TypeError(
                    f"Invalid weight type for key '{key}': {type(weights[key])}"
                )

        return 1

    def _diff(self, e, a, weight, weights, suppressed):
        suppressed = suppressed or is_suppressed(weights)
        t = type(e)
        if not isinstance(a, t):
            return TypesNotEqual(e, a, weight, suppressed).explain()
        if t is int:
            return self._int_diff(e, a, weight, suppressed)
        if t is str:
            return self._str_diff(e, a, weight, suppressed)
        if t is bool:
            return self._bool_diff(e, a, weight, suppressed)
        if t is float:
            return self._float_diff(e, a, weight, suppressed)
        if t is dict:
            return self._dict_diff(e, a, weight, weights, suppressed)
        if t is list:
            return self._list_diff(e, a, weight, weights, suppressed)
        return NO_DIFF

    def _calculate_similarity(self, e, a, weight, weights):
        diff = self._diff(e, a, weight, weights, False)
        result = self._create_result(diff, e, weight, weights)
        return result.similarity

    def _attributes_count(self, o):
        return self._weighted_attributes_count(o, 1, {})

    def _weighted_attributes_count(self, o, weight, weights):
        # Count the number of attributes in an object or list including nested objects and lists
        if isinstance(o, dict):
            sum = 0
            for k in o:
                k_weight = self._get_weight(weights, k) * weight
                nested_weights = self._get_nested_weights(weights, k)
                sum += self._weighted_attributes_count(o[k], k_weight, nested_weights)
            return sum
        elif isinstance(o, list):
            sum = 0
            nested_weights = self._get_nested_weights(weights, '_content')
            for i, v in enumerate(o):
                sum += self._weighted_attributes_count(v, weight, nested_weights)
            return sum
        else:
            return weight

    @classmethod
    def _int_diff(cls, e, a, weight, suppressed):
        if a == e:
            return NO_DIFF
        return ValuesNotEqual(e, a, weight, suppressed).explain()

    @classmethod
    def _bool_diff(cls, e, a, weight, suppressed):
        if a is e:
            return NO_DIFF
        return ValuesNotEqual(e, a, weight, suppressed).explain()

    @classmethod
    def _str_diff(cls, e, a, weight, suppressed):
        if a == e:
            return NO_DIFF
        return ValuesNotEqual(e, a, weight, suppressed).explain()

    def _float_diff(self, e, a, weight, suppressed):
        if a == e:
            return NO_DIFF
        if self._can_rounded_float():
            p = self._float_precision()
            e, a = round(e, p), round(a, p)
            if a == e:
                return NO_DIFF
        return ValuesNotEqual(e, a, weight, suppressed).explain()

    def _can_rounded_float(self):
        p = self._float_precision()
        return type(p) is int

    def _float_precision(self):
        path = 'types.float.allow_round'
        return self._config.get(path)

    def _get_nested_weights(self, weights, key):
        if (
            isinstance(weights, dict) and
            key in weights and
            isinstance(weights[key], dict)
        ):
            return weights.get(key)

        return {}

    def _dict_diff(self, e, a, dict_weight, weights, suppressed):
        missing_item_weight = self._get_weight(weights, '_missing')
        boost_missing_item_weight = get_boolean(weights, '_boost_missing')
        extra_item_weight = self._get_weight(weights, '_extra')
        boost_extra_item_weight = get_boolean(weights, '_boost_extra')

        d = {}

        for k in e:
            k_attr_weight = self._get_weight(weights, k)
            nested_weights = self._get_nested_weights(weights, k)
            k_suppressed = suppressed or is_suppressed(nested_weights)
            if k not in a:
                k_boost_weight = self._get_boost_weight(e[k], nested_weights) if boost_missing_item_weight else 1
                k_weight = dict_weight * k_attr_weight * missing_item_weight * k_boost_weight
                d[k] = KeyNotExist(k, None, k_weight, suppressed).explain()
            else:
                k_weight = dict_weight * k_attr_weight
                d[k] = self._diff(e[k], a[k], k_weight, nested_weights, suppressed)

        for k in a:
            k_attr_weight = self._get_weight(weights, k)
            nested_weights = self._get_nested_weights(weights, k)
            k_suppressed = suppressed or is_suppressed(nested_weights)
            if k not in e:
                k_boost_weight = self._get_boost_weight(a[k], nested_weights) if boost_extra_item_weight else 1
                k_weight = dict_weight * k_attr_weight * extra_item_weight * k_boost_weight
                d[k] = UnexpectedKey(None, k, k_weight, suppressed).explain()
            else:
                k_weight = dict_weight * k_attr_weight
                d[k] = self._diff(e[k], a[k], k_weight, nested_weights, suppressed)

        return self._without_empties(d)

    def _list_diff(self, e, a, weight, weights, suppressed):
        d = {}

        if self._need_compare_length():
            length_weight = self._get_weight(weights, '_length')
            d['_length'] = self._list_len_diff(e, a, weight * length_weight, suppressed)

        d['_content'] = self._list_content_diff_new(e, a, weight, weights, suppressed)

        return self._without_empties(d)

    def _need_compare_length(self):
        path = 'types.list.check_length'
        return self._config.get(path) is True

    def _length_diff_penalty(self, e, a):
        path = 'types.list.length_diff_penalty'
        return self._config.get(path) is True

    def _get_boost_weight(self, item, weights):
        diff = self._diff(item, {}, 1, weights, False)
        result = self._create_result(diff, item, 1, weights)
        return result.weighted_count

    def _list_content_diff_new(self, e, a, list_weight, weights, suppressed):
        content_weights = self._get_nested_weights(weights, '_content')
        missing_item_weight = self._get_weight(weights, '_missing')
        boost_missing_item_weight = get_boolean(weights, '_boost_missing')
        extra_item_weight = self._get_weight(weights, '_extra')
        boost_extra_item_weight = get_boolean(weights, '_boost_extra')
        pairing_threshold = weights['_pairing_threshold'] if '_pairing_threshold' in weights else 0.0

        result = {}

        if len(e) > 0 and len(a) > 0:
            # Prepare the score matrix for matrix in size len(e) x len(a)
            score_matrix = np.zeros((len(e), len(a)))

            # Calculate score for each pair of elements
            for i, v in enumerate(e):
                for j, w in enumerate(a):
                    if type(v) is type(w):
                        similarity = self._calculate_similarity(v, w, list_weight, content_weights)
                        score_matrix[i, j] = similarity

            # Hungarian algorithm optimizes the cost, so we need to convert scores to costs.
            # The cost is calculated as the maximum score minus the score.
            cost_matrix = score_matrix.max() - score_matrix

            # Using Hungarian algorithm (solving the minimization problem)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            pairs = list(zip(row_ind, col_ind))

            # Pairing (debug print)
            # print("Pairing:")
            # for v, w in zip(row_ind, col_ind):
            #     print(f"A{v+1} -> B{w+1} (score: {score_matrix[v, w]})")

            # Filter out pairs with a score below the threshold
            filtered_pairs = [(int(i), int(j)) for i, j in pairs if score_matrix[i, j] >= pairing_threshold] # also converts numpy.int64 to int

        else:
            # If one of the lists is empty, we cannot pair anything
            # This condition is necessary to avoid errors in the case of empty lists
            filtered_pairs = []

        # After pairing, we need to find the elements that were not matched
        # and add them to the result
        paired_e_items = [i for i, j in filtered_pairs]
        paired_a_items = [j for i, j in filtered_pairs]

        for i in range(len(e)):
            if i not in paired_e_items:
                i_boost_weight = self._get_boost_weight(e[i], content_weights) if boost_missing_item_weight else 1
                i_weight = list_weight * missing_item_weight * i_boost_weight
                result[i] = MissingListItem(e[i], None, i_weight, suppressed).explain()

        for j in range(len(a)):
            if j not in paired_a_items:
                j_boost_weight = self._get_boost_weight(a[j], content_weights) if boost_extra_item_weight else 1
                j_weight = list_weight * extra_item_weight * j_boost_weight
                result['extra_' + str(j)] = ExtraListItem(None, a[j], j_weight, suppressed).explain()

        # Now we need to check the elements that were matched
        for i, j in filtered_pairs:
            diff = self._diff(e[i], a[j], list_weight, content_weights, suppressed)
            if diff != NO_DIFF:
                result[i] = diff

        return result

    # def _list_content_diff(self, e, a, weight, weights):
    #     d = {}
    #     items_weights = weights.get('_list', {})
    #
    #     for i, v in enumerate(e):
    #         if v in a:
    #             continue
    #
    #         i_weight = self._get_weight(items_weights, i) * weight
    #         t = type(v)
    #
    #         if t in (int, str, bool, float):
    #             w = 0 if self._need_compare_length() else i_weight
    #             d[i] = ValueNotFound(v, None, w).explain()
    #         elif t is dict:
    #             d[i] = self._min_diff(v, a, self._dict_diff, i_weight, items_weights)
    #         elif t is list:
    #             d[i] = self._max_diff(v, a, self._list_diff, i_weight, items_weights)
    #     return self._without_empties(d)

    @classmethod
    def _max_diff(cls, e, lst, method, weight, weights):
        t = type(e)
        d = method(e, t(), weight, weights)
        for i, v in enumerate(lst):
            if type(v) is t:
                dd = method(e, v, weight, weights)
                if len(dd) <= len(d):
                    d = dd
        return d

    @classmethod
    def _min_diff(cls, e, lst, method, weight, weights):
        t = type(e)
        d = method(e, t(), weight, weights)
        for i, v in enumerate(lst):
            if type(v) is t:
                dd = method(e, v, weight, weights)
                if len(dd) <= len(d):
                    d = dd
                    break
        return d

    def _list_len_diff(self, e, a, weight, suppressed):
        e, a = len(e), len(a)

        if e == a:
            return NO_DIFF

        if self._length_diff_penalty(e, a):
            length_diff = abs(e - a)
            list_weight = weight * length_diff
        else:
            list_weight = weight

        return LengthsNotEqual(e, a, list_weight, suppressed).explain()

    @classmethod
    def _without_empties(cls, d):
        return {k: d[k] for k in d if d[k] != NO_DIFF}

    def report(self, diff):
        if self._need_write_to_console():
            self._write_to_console(diff)
        if self._need_write_to_file():
            self._write_to_file(diff)

    @classmethod
    def _write_to_console(cls, d):
        msg = json.dumps(d, indent=4)
        print(msg)

    def _write_to_file(self, d):
        config = self._config.get('output.file')
        with open(config.pop('name'), 'w') as fp:
            json.dump(d, fp, **config)

    def _need_write_to_console(self):
        path = 'output.console'
        return self._config.get(path) is True

    def _need_write_to_file(self):
        path = 'output.file.name'
        file_name = self._config.get(path)
        return type(file_name) is str

    def prepare(self, x):
        x = copy.deepcopy(x)
        return Ignore.transform(x, self._rules)
