from abc import ABC


class Error(ABC):
    expected = None
    received = None
    type = None  # New attribute to indicate the error type
    weight = 1
    suppress = False

    template = 'Expected: <{e}>, received: <{r}>'

    def __init__(self, expected, received, weight=1, suppress=False):
        self.expected = expected
        self.received = received
        self.weight = weight
        self.suppress = suppress
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
            '_suppress': self.suppress
        }


class TypesNotEqual(Error):
    template = 'Types not equal. Expected: <{e}>, received: <{r}>'

    def __init__(self, e, a, weight=1, suppress=False):
        e = type(e).__name__
        a = type(a).__name__
        super().__init__(e, a, weight, suppress)


class ValuesNotEqual(Error):
    template = 'Values not equal. Expected: <{e}>, received: <{r}>'


class KeyNotExist(Error):
    template = 'Key does not exist. Expected: <{e}>'


class LengthsNotEqual(Error):
    template = 'Lengths not equal. Expected <{e}>, received: <{r}>'


class ValueNotFound(Error):
    template = 'Value not found. Expected <{e}>'


class UnexpectedKey(Error):
    template = 'Unexpected key. Received: <{r}>'


class MissingListItem(Error):
    template = 'Missing list item. Expected <{e}>'


class ExtraListItem(Error):
    template = 'Extra list item. Received <{r}>'
