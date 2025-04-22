from abc import ABC


class Error(ABC):
    expected = None
    received = None
    type = None  # New attribute to indicate the error type

    template = 'Expected: <{e}>, received: <{r}>'

    def __init__(self, expected, received):
        self.expected = expected
        self.received = received
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
            '_error': self.type,  # Include the type in the explanation
        }


class TypesNotEqual(Error):
    template = 'Types not equal. Expected: <{e}>, received: <{r}>'

    def __init__(self, e, a):
        e = type(e).__name__
        a = type(a).__name__
        super().__init__(e, a)


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
