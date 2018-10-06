from abc import ABC

from jsoncompare.utils import cls_name


class Error(ABC):
    expected = None
    received = None

    template = 'Expected: <{e}>, received: <{r}>'

    def __init__(self, expected, received):
        self.expected = expected
        self.received = received

    @property
    def message(self):
        msg = self.template.format(e=self.expected, r=self.received)
        return msg

    def to_dict(self):
        return {
            '_message': self.message,
            '_expected': self.expected,
            '_received': self.received,
        }


class TypesNotEqual(Error):
    template = 'Types not equal. Expected: <{e}>, received: <{r}>'

    def __init__(self, expected, actual):
        e = cls_name(expected)
        a = cls_name(actual)
        super().__init__(e, a)


class ValuesNotEqual(Error):
    template = 'Values not equal. Expected: <{e}>, received: <{r}>'


class KeyNotExist(Error):
    template = 'Key does not exists. Expected: <{e}>'
