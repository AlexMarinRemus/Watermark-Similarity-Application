import pytest


class TestMain:
    def test_main(self):
        assert True, "This should pass, but if it doesn't this string is shown in the error message"

    def test_raise_error(self):
        with pytest.raises(ZeroDivisionError):
            1/0


if __name__ == '__main__':
    pytest.main()  # Run this file with pytest
