import pytest

from sdgx.models.components.utils import StrValuedBaseEnum as SE
class A(SE):
    a = "1"
    b = "2"
def test_se():
    a = A("1")
    b = A("2")
    assert isinstance(a.value, str) and a.value == "1"
    assert b.value == "2"
    assert A.values == {"1", "2"}
    assert a != b
    assert ['1', '2', '3'] not in A
    assert '1' in A
    assert ['2'] in A
    assert ['1', '2'] in A
    assert '1' == a
    assert 1 != a