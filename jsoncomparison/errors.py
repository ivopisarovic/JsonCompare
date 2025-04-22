from abc import ABC


class Error(ABC):
    expected = None
    received = None
    type = None  # New attribute to indicate the error type
    weight = 1

    template = 'Expected: <{e}>, received: <{r}>'

    def __init__(self, expected, received, weight=1):
        self.expected = expected
        self.received = received
        self.weight = weight
        self.type = self.__class__.__name__  # Set the type to the subclass name

    @property
    def message(self):
        msg = self.template.format(e=self.expected, r=self.received)
        return msg

    def explain(self):
        return {
            '_message': self.message,
            '_expected': self.expected,
            '_received': self.received,
            '_error': self.type,
            '_weight': self.weight,
        }


class TypesNotEqual(Error):
    template = 'Types not equal. Expected: <{e}>, received: <{r}>'

    def __init__(self, e, a, weight):
        e = type(e).__name__
        a = type(a).__name__
        super().__init__(e, a, weight)


class ValuesNotEqual(Error):
    template = 'Values not equal. Expected: <{e}>, received: <{r}>'


class KeyNotExist(Error):
    template = 'Key does not exist. Expected: <{e}>'


class LengthsNotEqual(Error):
    template = 'Lengths not equal. Expected <{e}>, received: <{r}>'
    diff = None

    def __init__(self, expected_length, received_length, weight=1):
        self.diff = abs(expected_length - received_length)
        list_weight = weight * self.diff
        super().__init__(expected_length, received_length, list_weight)


class ValueNotFound(Error):
    template = 'Value not found. Expected <{e}>'

    def __init__(self, expected, received):
        super().__init__(expected, received, 0)


class UnexpectedKey(Error):
    template = 'Unexpected key. Received: <{r}>'
