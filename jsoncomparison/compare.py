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
    ValueNotFound,
    ValuesNotEqual, MissingListItem, ExtraListItem,
)
from .ignore import Ignore

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

    __slots__ = ("_failed", "_failed_weighted", "_count", "_diff")

    def __init__(self, expected, diff):
        self._diff = diff
        self._count = self._count_attributes_deep(expected)
        self._failed = self._count_failed(diff, False)
        self._failed_weighted = self._count_failed(diff, True)

    def _count_attributes_deep(self, o):
        # Count the number of attributes in an object or list including nested objects and lists
        if isinstance(o, dict):
            return sum(self._count_attributes_deep(v) for v in o.values())
        elif isinstance(o, list):
            return sum(self._count_attributes_deep(v) for v in o)
        else:
            return 1

    def _count_failed(self, d, weighted):
        if self._is_problem(d):
            if weighted and '_weight' in d:
                return d['_weight']
            else:
                return 1
        else:
            return sum(self._count_failed(v, weighted) for v in d.values())

    def _is_problem(self, d):
        if isinstance(d, dict) and '_error' in d:
            return True
        return False

    @property
    def failed(self):
        return self._failed

    @property
    def failed_weighted(self):
        return self._failed_weighted

    @property
    def count(self):
        return self._count

    @property
    def similarity(self):
        if self._count == 0:
            return 0
        return (self._count - self._failed_weighted) / self._count

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
        weight = self._weights['_weight'] if '_weight' in self._weights else 1
        diff = self._diff(e, a, weight, self._weights)
        self.report(diff)
        return diff

    def calculate_score(self, expected, actual):
        diff = self.check(expected, actual)
        return Result(expected, diff)

    @staticmethod
    def _get_weight(weights, key):
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

    def _diff(self, e, a, weight, weights):
        t = type(e)
        if not isinstance(a, t):
            return TypesNotEqual(e, a, weight).explain()
        if t is int:
            return self._int_diff(e, a, weight)
        if t is str:
            return self._str_diff(e, a, weight)
        if t is bool:
            return self._bool_diff(e, a, weight)
        if t is float:
            return self._float_diff(e, a, weight)
        if t is dict:
            return self._dict_diff(e, a, weight, weights)
        if t is list:
            return self._list_diff(e, a, weight, weights)
        return NO_DIFF

    def _calculate_similarity(self, e, a, weight, weights):
        diff = self._diff(e, a, weight, weights)
        return Result(e, diff).similarity

    @classmethod
    def _int_diff(cls, e, a, weight):
        if a == e:
            return NO_DIFF
        return ValuesNotEqual(e, a, weight).explain()

    @classmethod
    def _bool_diff(cls, e, a, weight):
        if a is e:
            return NO_DIFF
        return ValuesNotEqual(e, a, weight).explain()

    @classmethod
    def _str_diff(cls, e, a, weight):
        if a == e:
            return NO_DIFF
        return ValuesNotEqual(e, a, weight).explain()

    def _float_diff(self, e, a, weight):
        if a == e:
            return NO_DIFF
        if self._can_rounded_float():
            p = self._float_precision()
            e, a = round(e, p), round(a, p)
            if a == e:
                return NO_DIFF
        return ValuesNotEqual(e, a, weight).explain()

    def _can_rounded_float(self):
        p = self._float_precision()
        return type(p) is int

    def _float_precision(self):
        path = 'types.float.allow_round'
        return self._config.get(path)

    def _dict_diff(self, e, a, weight, weights):
        d = {}
        for k in e:
            k_weight = self._get_weight(weights, k) * weight
            if k not in a:
                d[k] = KeyNotExist(k, None, k_weight).explain()
            else:
                nested_weights = weights.get(k, {})
                d[k] = self._diff(e[k], a[k], k_weight, nested_weights)

        for k in a:
            k_weight = self._get_weight(weights, k) * weight
            if k not in e:
                d[k] = UnexpectedKey(None, k, k_weight).explain()
            else:
                nested_weights = weights.get(k, {})
                d[k] = self._diff(e[k], a[k], k_weight, nested_weights)

        return self._without_empties(d)

    def _list_diff(self, e, a, weight, weights):
        d = {}

        if self._need_compare_length():
            length_weight = self._get_weight(weights, '_length')
            d['_length'] = self._list_len_diff(e, a, weight * length_weight)

        d['_content'] = self._list_content_diff_new(e, a, weight, weights)

        return self._without_empties(d)

    def _need_compare_length(self):
        path = 'types.list.check_length'
        return self._config.get(path) is True

    def _length_diff_penalty(self, e, a):
        path = 'types.list.length_diff_penalty'
        return self._config.get(path) is True

    def _list_content_diff_new(self, e, a, list_weight, weights):
        content_weights = weights.get('_content', {})
        missing_item_weight = self._get_weight(weights, '_missing')
        extra_item_weight = self._get_weight(weights, '_extra')

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

        # Pairing (debug print)
        # print("Pairing:")
        # for v, w in zip(row_ind, col_ind):
        #     print(f"A{v+1} -> B{w+1} (score: {score_matrix[v, w]})")

        # # Final pairing score (debug print)
        # total_score = score_matrix[row_ind, col_ind].sum()
        # print(f"Final score: {total_score}")

        result = {}

        # After pairing, we need to find the elements that were not matched
        # and add them to the result
        for i in range(len(e)):
            if i not in row_ind:
                result[i] = MissingListItem(e[i], None, list_weight * missing_item_weight).explain()

        for j in range(len(a)):
            if j not in col_ind:
                result['extra_' + str(j)] = ExtraListItem(None, a[j], list_weight * extra_item_weight).explain()

        # Now we need to check the elements that were matched
        for i, j in zip(row_ind, col_ind):
            diff = self._diff(e[i], a[j], list_weight, content_weights)
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

    def _list_len_diff(self, e, a, weight):
        e, a = len(e), len(a)

        if e == a:
            return NO_DIFF

        if self._length_diff_penalty(e, a):
            length_diff = abs(e - a)
            list_weight = weight * length_diff
        else:
            list_weight = weight

        return LengthsNotEqual(e, a, list_weight).explain()

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
