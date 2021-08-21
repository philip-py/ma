import pytest
import os
from app.src.d00_utils import add_1, sub_1, divide_by_zero
# from app import add_1, sub_1

def test_adding():
    assert add_1(2) == 3

def test_subtracting():
    assert sub_1(2) == 1

# @pytest.mark.smoke
def test_expecting_exception():
    with pytest.raises(ZeroDivisionError) as excinfo:
        divide_by_zero(100)
    assert excinfo.value.args[0] == 'division by zero'

