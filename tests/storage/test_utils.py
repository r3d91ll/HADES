"""Unit tests for arango naming helpers."""
from src.storage.arango.utils import safe_name, safe_key


def test_safe_name_basic():
    assert safe_name("MyCollection") == "MyCollection"


def test_safe_name_strip_bad_chars():
    assert safe_name("my coll$#") == "my_coll__"


def test_safe_name_no_leading_underscore():
    assert not safe_name("__bad").startswith("_")


def test_safe_key_spaces_removed():
    assert safe_key("hello world") == "hello_world"


def test_safe_key_remove_bad():
    assert safe_key("key/with?bad") == "keywithbad"
