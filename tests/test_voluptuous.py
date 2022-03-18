import copy
import collections
from enum import Enum
import os
import sys

import pytest

from voluptuous import (
    Schema, Required, Exclusive, Inclusive, Optional, Extra, Invalid, In, Remove,
    Literal, Url, MultipleInvalid, LiteralInvalid, TypeInvalid, NotIn, Match, Email,
    Replace, Range, Coerce, All, Any, Length, FqdnUrl, ALLOW_EXTRA, PREVENT_EXTRA,
    validate, ExactSequence, Equal, Unordered, Number, Maybe, Datetime, Date,
    Contains, Marker, IsDir, IsFile, PathExists, SomeOf, TooManyValid, Self,
    raises, Union, Clamp)
from voluptuous.humanize import humanize_error
from voluptuous.util import Capitalize, Lower, Strip, Title, Upper


def test_new_required_test():
    schema = Schema({
        'my_key': All(int, Range(1, 20)),
    }, required=True)
    assert schema.required


def test_exact_sequence():
    schema = Schema(ExactSequence([int, int]))
    with pytest.raises(Invalid):
        schema([1, 2, 3])
    assert schema([1, 2]) == [1, 2]


def test_required():
    """Verify that Required works."""
    schema = Schema({Required('q'): int})
    schema({"q": 123})
    with pytest.raises(Invalid) as e:
        schema({})
    assert str(e.value) == "required key not provided @ data['q']"


def test_extra_with_required():
    """Verify that Required does not break Extra."""
    schema = Schema({Required('toaster'): str, Extra: object})
    r = schema({'toaster': 'blue', 'another_valid_key': 'another_valid_value'})
    assert r == {'toaster': 'blue', 'another_valid_key': 'another_valid_value'}


def test_iterate_candidates():
    """Verify that the order for iterating over mapping candidates is right."""
    schema = {
        "toaster": str,
        Extra: object,
    }
    # toaster should be first.
    from voluptuous.schema_builder import _iterate_mapping_candidates
    assert _iterate_mapping_candidates(schema)[0][0] == 'toaster'


def test_in():
    """Verify that In works."""
    schema = Schema({"color": In(frozenset(["red", "blue", "yellow"]))})
    schema({"color": "blue"})
    with pytest.raises(Invalid) as e:
        schema({"color": "orange"})
    assert str(e.value) == "value must be one of ['blue', 'red', 'yellow'] for dictionary value @ data['color']"


def test_not_in():
    """Verify that NotIn works."""
    schema = Schema({"color": NotIn(frozenset(["red", "blue", "yellow"]))})
    schema({"color": "orange"})
    with pytest.raises(Invalid) as e:
        schema({"color": "blue"})
    assert str(e.value) == "value must not be one of ['blue', 'red', 'yellow'] for dictionary value @ data['color']"


def test_contains():
    """Verify contains validation method."""
    schema = Schema({'color': Contains('red')})
    schema({'color': ['blue', 'red', 'yellow']})
    with pytest.raises(Invalid) as e:
        schema({'color': ['blue', 'yellow']})
    assert str(e.value) == "value is not allowed for dictionary value @ data['color']"


def test_remove():
    """Verify that Remove works."""
    # remove dict keys
    schema = Schema({"weight": int,
                     Remove("color"): str,
                     Remove("amount"): int})
    out_ = schema({"weight": 10, "color": "red", "amount": 1})
    assert "color" not in out_ and "amount" not in out_

    # remove keys by type
    schema = Schema({"weight": float,
                     "amount": int,
                     # remvove str keys with int values
                     Remove(str): int,
                     # keep str keys with str values
                     str: str})
    out_ = schema({"weight": 73.4,
                   "condition": "new",
                   "amount": 5,
                   "left": 2})
    # amount should stay since it's defined
    # other string keys with int values will be removed
    assert "amount" in out_ and "left" not in out_
    # string keys with string values will stay
    assert "condition" in out_

    # remove value from list
    schema = Schema([Remove(1), int])
    out_ = schema([1, 2, 3, 4, 1, 5, 6, 1, 1, 1])
    assert out_ == [2, 3, 4, 5, 6]

    # remove values from list by type
    schema = Schema([1.0, Remove(float), int])
    out_ = schema([1, 2, 1.0, 2.0, 3.0, 4])
    assert out_ == [1, 2, 1.0, 4]


def test_extra_empty_errors():
    schema = Schema({'a': {Extra: object}}, required=True)
    schema({'a': {}})


def test_literal():
    """ Test with Literal """

    schema = Schema([Literal({"a": 1}), Literal({"b": 1})])
    schema([{"a": 1}])
    schema([{"b": 1}])
    schema([{"a": 1}, {"b": 1}])

    with pytest.raises(Invalid) as e:
        schema([{"c": 1}])
    assert str(e.value) == "{'c': 1} not match for {'b': 1} @ data[0]"

    schema = Schema(Literal({"a": 1}))
    with pytest.raises(MultipleInvalid) as e:
        schema({"b": 1})
    assert str(e.value) == "{'b': 1} not match for {'a': 1}"
    assert len(e.value.errors) == 1
    assert type(e.value.errors[0]) == LiteralInvalid


def test_class():
    class C1:
        pass

    schema = Schema(C1)
    schema(C1())

    with pytest.raises(MultipleInvalid) as e:
        schema(None)
    assert str(e.value) == "expected C1"
    assert len(e.value.errors) == 1
    assert type(e.value.errors[0]) == TypeInvalid


def test_email_validation():
    """ Test with valid email address """
    schema = Schema({"email": Email()})
    out_ = schema({"email": "example@example.com"})

    assert 'example@example.com' == out_["email"]


def test_email_validation_with_none():
    """ Test with invalid None email address """
    schema = Schema({"email": Email()})
    with pytest.raises(MultipleInvalid) as e:
        schema({"email": None})
    assert str(e.value) == "expected an email address for dictionary value @ data['email']"


def test_email_validation_with_empty_string():
    """ Test with empty string email address"""
    schema = Schema({"email": Email()})
    with pytest.raises(MultipleInvalid) as e:
        schema({"email": ''})
    assert str(e.value) == "expected an email address for dictionary value @ data['email']"


def test_email_validation_without_host():
    """ Test with empty host name in email address """
    schema = Schema({"email": Email()})
    with pytest.raises(MultipleInvalid) as e:
        schema({"email": 'a@.com'})
    assert str(e.value) == "expected an email address for dictionary value @ data['email']"


def test_fqdn_url_validation():
    """ Test with valid fully qualified domain name URL """
    schema = Schema({"url": FqdnUrl()})
    out_ = schema({"url": "http://example.com/"})

    assert 'http://example.com/' == out_["url"]


def test_fqdn_url_without_domain_name():
    """ Test with invalid fully qualified domain name URL """
    schema = Schema({"url": FqdnUrl()})
    with pytest.raises(MultipleInvalid) as e:
        schema({"url": "http://localhost/"})
    assert str(e.value) == "expected a fully qualified domain name URL for dictionary value @ data['url']"


def test_fqdnurl_validation_with_none():
    """ Test with invalid None FQDN URL """
    schema = Schema({"url": FqdnUrl()})
    with pytest.raises(MultipleInvalid) as e:
        schema({"url": None})
    assert str(e.value) == "expected a fully qualified domain name URL for dictionary value @ data['url']"


def test_fqdnurl_validation_with_empty_string():
    """ Test with empty string FQDN URL """
    schema = Schema({"url": FqdnUrl()})
    with pytest.raises(MultipleInvalid) as e:
        schema({"url": ''})
    assert str(e.value) == "expected a fully qualified domain name URL for dictionary value @ data['url']"


def test_fqdnurl_validation_without_host():
    """ Test with empty host FQDN URL """
    schema = Schema({"url": FqdnUrl()})
    with pytest.raises(MultipleInvalid) as e:
        schema({"url": 'http://'})
    assert str(e.value) == "expected a fully qualified domain name URL for dictionary value @ data['url']"


def test_url_validation():
    """ Test with valid URL """
    schema = Schema({"url": Url()})
    out_ = schema({"url": "http://example.com/"})

    assert 'http://example.com/' == out_["url"]


def test_url_validation_with_none():
    """ Test with invalid None URL"""
    schema = Schema({"url": Url()})
    with pytest.raises(MultipleInvalid) as e:
        schema({"url": None})
    assert str(e.value) == "expected a URL for dictionary value @ data['url']"


def test_url_validation_with_empty_string():
    """ Test with empty string URL """
    schema = Schema({"url": Url()})
    with pytest.raises(MultipleInvalid) as e:
        schema({"url": ''})
    assert str(e.value) == "expected a URL for dictionary value @ data['url']"


def test_url_validation_without_host():
    """ Test with empty host URL """
    schema = Schema({"url": Url()})
    with pytest.raises(MultipleInvalid) as e:
        schema({"url": 'http://'})
    assert str(e.value) == "expected a URL for dictionary value @ data['url']"


def test_copy_dict_undefined():
    """ Test with a copied dictionary """
    fields = {
        Required("foo"): int
    }
    copied_fields = copy.deepcopy(fields)

    schema = Schema(copied_fields)

    # This used to raise a `TypeError` because the instance of `Undefined`
    # was a copy, so object comparison would not work correctly.
    with pytest.raises(MultipleInvalid):
        schema({"foo": "bar"})


def test_sorting():
    """ Expect alphabetic sorting """
    foo = Required('foo')
    bar = Required('bar')
    items = [foo, bar]
    result = sorted(items)
    assert result == [bar, foo]


def test_schema_extend():
    """Verify that Schema.extend copies schema keys from both."""

    base = Schema({'a': int}, required=True)
    extension = {'b': str}
    extended = base.extend(extension)

    assert base.schema == {'a': int}
    assert extension == {'b': str}
    assert extended.schema == {'a': int, 'b': str}
    assert extended.required == base.required
    assert extended.extra == base.extra
    assert isinstance(extended, Schema)


def test_schema_extend_overrides():
    """Verify that Schema.extend can override required/extra parameters."""
    base = Schema({'a': int}, required=True)
    extended = base.extend({'b': str}, required=False, extra=ALLOW_EXTRA)

    assert base.required is True
    assert base.extra == PREVENT_EXTRA
    assert extended.required is False
    assert extended.extra == ALLOW_EXTRA


def test_schema_extend_key_swap():
    """Verify that Schema.extend can replace keys, even when different markers are used"""
    base = Schema({Optional('a'): int})
    extension = {Required('a'): int}
    extended = base.extend(extension)

    assert len(base.schema) == 1
    assert isinstance(list(base.schema)[0], Optional)
    assert len(extended.schema) == 1
    assert isinstance(list(extended.schema)[0], Required)


def test_subschema_extension():
    """Verify that Schema.extend adds and replaces keys in a subschema"""
    base = Schema({'a': {'b': int, 'c': float}})
    extension = {'d': str, 'a': {'b': str, 'e': int}}
    extended = base.extend(extension)

    assert base.schema == {'a': {'b': int, 'c': float}}
    assert extension == {'d': str, 'a': {'b': str, 'e': int}}
    assert extended.schema == {'a': {'b': str, 'c': float, 'e': int}, 'd': str}


def test_schema_extend_handles_schema_subclass():
    """Verify that Schema.extend handles a subclass of Schema"""
    class S(Schema):
        pass

    base = S({Required('a'): int})
    extension = {Optional('b'): str}
    extended = base.extend(extension)

    expected_schema = {Required('a'): int, Optional('b'): str}
    assert extended.schema == expected_schema
    assert isinstance(extended, S)


def test_equality():
    assert Schema('foo') == Schema('foo')

    assert Schema(['foo', 'bar', 'baz']) == Schema(['foo', 'bar', 'baz'])

    # Ensure two Schemas w/ two equivalent dicts initialized in a different
    # order are considered equal.
    dict_a = {}
    dict_a['foo'] = 1
    dict_a['bar'] = 2
    dict_a['baz'] = 3

    dict_b = {}
    dict_b['baz'] = 3
    dict_b['bar'] = 2
    dict_b['foo'] = 1

    assert Schema(dict_a), Schema(dict_b)


def test_equality_negative():
    """Verify that Schema objects are not equal to string representations"""
    assert not Schema('foo') == 'foo'

    assert not Schema(['foo', 'bar']) == "['foo', 'bar']"
    assert not Schema(['foo', 'bar']) == Schema("['foo', 'bar']")

    assert not Schema({'foo': 1, 'bar': 2}) == "{'foo': 1, 'bar': 2}"
    assert not Schema({'foo': 1, 'bar': 2}) == Schema("{'foo': 1, 'bar': 2}")


def test_inequality():
    assert Schema('foo') != 'foo'

    assert Schema(['foo', 'bar']) != "['foo', 'bar']"
    assert Schema(['foo', 'bar']) != Schema("['foo', 'bar']")

    assert Schema({'foo': 1, 'bar': 2}) != "{'foo': 1, 'bar': 2}"
    assert Schema({'foo': 1, 'bar': 2}) != Schema("{'foo': 1, 'bar': 2}")


def test_inequality_negative():
    assert not Schema('foo') != Schema('foo')

    assert not Schema(['foo', 'bar', 'baz']) != Schema(['foo', 'bar', 'baz'])

    # Ensure two Schemas w/ two equivalent dicts initialized in a different
    # order are considered equal.
    dict_a = {}
    dict_a['foo'] = 1
    dict_a['bar'] = 2
    dict_a['baz'] = 3

    dict_b = {}
    dict_b['baz'] = 3
    dict_b['bar'] = 2
    dict_b['foo'] = 1

    assert not Schema(dict_a) != Schema(dict_b)


def test_repr():
    """Verify that __repr__ returns valid Python expressions"""
    match = Match('a pattern', msg='message')
    replace = Replace('you', 'I', msg='you and I')
    range_ = Range(min=0, max=42, min_included=False,
                   max_included=False, msg='number not in range')
    coerce_ = Coerce(int, msg="moo")
    all_ = All('10', Coerce(int), msg='all msg')
    maybe_int = Maybe(int)

    assert repr(match) == "Match('a pattern', msg='message')"
    assert repr(replace) == "Replace('you', 'I', msg='you and I')"
    assert repr(range_) == "Range(min=0, max=42, min_included=False, max_included=False, msg='number not in range')"

    assert repr(coerce_) == "Coerce(int, msg='moo')"
    assert repr(all_) == "All('10', Coerce(int, msg=None), msg='all msg')"
    assert repr(maybe_int) == "Any(None, %s, msg=None)" % str(int)


def test_list_validation_messages():
    """ Make sure useful error messages are available """

    def is_even(value):
        if value % 2:
            raise Invalid('%i is not even' % value)
        return value

    schema = Schema(dict(even_numbers=[All(int, is_even)]))

    with pytest.raises(MultipleInvalid) as e:
        schema(dict(even_numbers=[3]))
    assert len(e.value.errors) == 1
    assert str(e.value.errors[0]) == "3 is not even @ data['even_numbers'][0]"
    assert str(e.value) == "3 is not even @ data['even_numbers'][0]"


def test_nested_multiple_validation_errors():
    """ Make sure useful error messages are available """

    def is_even(value):
        if value % 2:
            raise Invalid('%i is not even' % value)
        return value

    schema = Schema(dict(even_numbers=All([All(int, is_even)],
                                          Length(min=1))))

    with pytest.raises(MultipleInvalid) as e:
        schema(dict(even_numbers=[3]))
    assert len(e.value.errors) == 1
    assert str(e.value.errors[0]) == "3 is not even @ data['even_numbers'][0]"
    assert str(e.value) == "3 is not even @ data['even_numbers'][0]"


def test_humanize_error():
    data = {
        'a': 'not an int',
        'b': [123]
    }
    schema = Schema({
        'a': int,
        'b': [str]
    })
    with pytest.raises(MultipleInvalid) as e:
        schema(data)
    assert (
        humanize_error(data, e.value) ==
        "expected int for dictionary value @ data['a']. Got 'not an int'\n"
        "expected str @ data['b'][0]. Got 123"
    )


def test_fix_157():
    s = Schema(All([Any('one', 'two', 'three')]), Length(min=1))
    assert ['one'] == s(['one'])
    with pytest.raises(MultipleInvalid):
        s(['four'])


def test_range_inside():
    s = Schema(Range(min=0, max=10))
    assert 5 == s(5)


def test_range_outside():
    s = Schema(Range(min=0, max=10))
    with pytest.raises(MultipleInvalid):
        s(12)
    with pytest.raises(MultipleInvalid):
        s(-1)


def test_range_no_upper_limit():
    s = Schema(Range(min=0))
    assert 123 == s(123)
    with pytest.raises(MultipleInvalid):
        s(-1)


def test_range_no_lower_limit():
    s = Schema(Range(max=10))
    assert -1 == s(-1)
    with pytest.raises(MultipleInvalid):
        s(123)


def test_range_excludes_nan():
    s = Schema(Range(min=0, max=10))
    with pytest.raises(MultipleInvalid):
        s(float('nan'))


def test_range_excludes_none():
    s = Schema(Range(min=0, max=10))
    with pytest.raises(MultipleInvalid):
        s(None)


def test_range_excludes_string():
    s = Schema(Range(min=0, max=10))
    with pytest.raises(MultipleInvalid):
        s("abc")


def test_range_excludes_unordered_object():
    class MyObject(object):
        pass

    s = Schema(Range(min=0, max=10))
    with pytest.raises(MultipleInvalid):
        s(MyObject())


def test_clamp_inside():
    s = Schema(Clamp(min=1, max=10))
    assert 5 == s(5)


def test_clamp_above():
    s = Schema(Clamp(min=1, max=10))
    assert 10 == s(12)


def test_clamp_below():
    s = Schema(Clamp(min=1, max=10))
    assert 1 == s(-3)


def test_clamp_invalid():
    s = Schema(Clamp(min=1, max=10))
    with pytest.raises(MultipleInvalid):
        s(None)
    with pytest.raises(MultipleInvalid):
        s("abc")


def test_length_ok():
    v1 = ['a', 'b', 'c']
    s = Schema(Length(min=1, max=10))
    assert v1 == s(v1)
    v2 = "abcde"
    assert v2 == s(v2)


def test_length_too_short():
    v1 = []
    s = Schema(Length(min=1, max=10))
    with pytest.raises(MultipleInvalid):
        s(v1)
    v2 = ''
    with pytest.raises(MultipleInvalid):
        s(v2)


def test_length_too_long():
    v = ['a', 'b', 'c']
    s = Schema(Length(min=0, max=2))
    with pytest.raises(MultipleInvalid):
        s(v)


def test_length_invalid():
    v = None
    s = Schema(Length(min=0, max=2))
    with pytest.raises(MultipleInvalid):
        s(v)


def test_equal():
    s = Schema(Equal(1))
    s(1)
    with pytest.raises(Invalid):
        s(2)
    s = Schema(Equal('foo'))
    s('foo')
    with pytest.raises(Invalid):
        s('bar')
    s = Schema(Equal([1, 2]))
    s([1, 2])
    with pytest.raises(Invalid):
        s([])
    with pytest.raises(Invalid):
        s([1, 2, 3])
    # Evaluates exactly, not through validators
    s = Schema(Equal(str))
    with pytest.raises(Invalid):
        s('foo')


def test_unordered():
    # Any order is OK
    s = Schema(Unordered([2, 1]))
    s([2, 1])
    s([1, 2])
    # Amount of errors is OK
    with pytest.raises(Invalid):
        s([2, 0])
    with pytest.raises(MultipleInvalid):
        s([0, 0])
    # Different length is NOK
    with pytest.raises(Invalid):
        s([1])
    with pytest.raises(Invalid):
        s([1, 2, 0])
    with pytest.raises(MultipleInvalid):
        s([1, 2, 0, 0])
    # Other type than list or tuple is NOK
    with pytest.raises(Invalid):
        s('foo')
    with pytest.raises(Invalid):
        s(10)
    # Validators are evaluated through as schemas
    s = Schema(Unordered([int, str]))
    s([1, '2'])
    s(['1', 2])
    s = Schema(Unordered([{'foo': int}, []]))
    s([{'foo': 3}, []])
    # Most accurate validators must be positioned on left
    s = Schema(Unordered([int, 3]))
    with pytest.raises(Invalid):
        s([3, 2])
    s = Schema(Unordered([3, int]))
    s([3, 2])


def test_maybe():
    s = Schema(Maybe(int))
    assert s(1) == 1
    assert s(None) is None
    with pytest.raises(Invalid):
        s('foo')

    s = Schema(Maybe({str: Coerce(int)}))
    assert s({'foo': '100'}) == {'foo': 100}
    assert s(None) is None
    with pytest.raises(Invalid):
        s({'foo': 'bar'})


def test_maybe_accepts_msg():
    s = Schema(Maybe(int, msg='int or None expected'))
    with raises(MultipleInvalid, 'int or None expected'):
        assert s([])


def test_maybe_returns_default_error():
    schema = Schema(Maybe(Range(1, 2)))

    # The following should be valid
    schema(None)
    schema(1)
    schema(2)

    with pytest.raises(MultipleInvalid) as e:
        # Should trigger a MultipleInvalid exception
        schema(3)
    assert str(e.value) == "not a valid value"


def test_schema_empty_list():
    s = Schema([])
    s([])

    with pytest.raises(MultipleInvalid) as e:
        s([123])
    assert str(e.value) == "not a valid value @ data[123]"

    with pytest.raises(MultipleInvalid) as e:
        s({'var': 123})
    assert str(e.value) == "expected a list"


def test_schema_empty_dict():
    s = Schema({})
    s({})

    with pytest.raises(MultipleInvalid) as e:
        s({'var': 123})
    assert str(e.value) == "extra keys not allowed @ data['var']"

    with pytest.raises(MultipleInvalid) as e:
        s([123])
    assert str(e.value) == "expected a dictionary"


def test_schema_empty_dict_key():
    """ https://github.com/alecthomas/voluptuous/pull/434 """
    s = Schema({'var': []})
    s({'var': []})

    with pytest.raises(MultipleInvalid) as e:
        s({'var': [123]})
    assert str(e.value) == "not a valid value for dictionary value @ data['var']"


def test_schema_decorator_match_with_args():
    @validate(int)
    def fn(arg):
        return arg

    fn(1)


def test_schema_decorator_unmatch_with_args():
    @validate(int)
    def fn(arg):
        return arg

    with pytest.raises(Invalid):
        fn(1.0)


def test_schema_decorator_match_with_kwargs():
    @validate(arg=int)
    def fn(arg):
        return arg

    fn(1)


def test_schema_decorator_unmatch_with_kwargs():
    @validate(arg=int)
    def fn(arg):
        return arg

    with pytest.raises(Invalid):
        fn(1.0)


def test_schema_decorator_match_return_with_args():
    @validate(int, __return__=int)
    def fn(arg):
        return arg

    fn(1)


def test_schema_decorator_unmatch_return_with_args():
    @validate(int, __return__=int)
    def fn(arg):
        return "hello"

    with pytest.raises(Invalid):
        fn(1)


def test_schema_decorator_match_return_with_kwargs():
    @validate(arg=int, __return__=int)
    def fn(arg):
        return arg

    fn(1)


def test_schema_decorator_unmatch_return_with_kwargs():
    @validate(arg=int, __return__=int)
    def fn(arg):
        return "hello"

    with pytest.raises(Invalid):
        fn(1)


def test_schema_decorator_return_only_match():
    @validate(__return__=int)
    def fn(arg):
        return arg

    fn(1)


def test_schema_decorator_return_only_unmatch():
    @validate(__return__=int)
    def fn(arg):
        return "hello"

    with pytest.raises(Invalid):
        fn(1)


def test_schema_decorator_partial_match_called_with_args():
    @validate(arg1=int)
    def fn(arg1, arg2):
        return arg1

    fn(1, "foo")


def test_schema_decorator_partial_unmatch_called_with_args():
    @validate(arg1=int)
    def fn(arg1, arg2):
        return arg1

    with pytest.raises(Invalid):
        fn('foo', 'bar')


def test_schema_decorator_partial_match_called_with_kwargs():
    @validate(arg2=int)
    def fn(arg1, arg2):
        return arg1

    fn(arg1="foo", arg2=1)


def test_schema_decorator_partial_unmatch_called_with_kwargs():
    @validate(arg2=int)
    def fn(arg1, arg2):
        return arg1

    with pytest.raises(Invalid):
        fn(arg1=1, arg2="foo")


def test_unicode_as_key():
    schema = Schema({str: int})
    schema({"foobar": 1})


def test_number_validation_with_string():
    """ Test with Number with string"""
    schema = Schema({"number": Number(precision=6, scale=2)})
    with pytest.raises(MultipleInvalid) as e:
        schema({"number": 'teststr'})
    assert str(e.value) == "Value must be a number enclosed with string for dictionary value @ data['number']"


def test_number_validation_with_invalid_precision_invalid_scale():
    """ Test with Number with invalid precision and scale"""
    schema = Schema({"number": Number(precision=6, scale=2)})
    with pytest.raises(MultipleInvalid) as e:
        schema({"number": '123456.712'})
    assert str(e.value) == "Precision must be equal to 6, and Scale must be equal to 2 for dictionary value @ data['number']"


def test_number_validation_with_valid_precision_scale_yield_decimal_true():
    """ Test with Number with valid precision and scale"""
    schema = Schema({"number": Number(precision=6, scale=2, yield_decimal=True)})
    out_ = schema({"number": '1234.00'})
    assert float(out_.get("number")) == 1234.00


def test_number_when_precision_scale_none_yield_decimal_true():
    """ Test with Number with no precision and scale"""
    schema = Schema({"number": Number(yield_decimal=True)})
    out_ = schema({"number": '12345678901234'})
    assert out_.get("number") == 12345678901234


def test_number_when_precision_none_n_valid_scale_case1_yield_decimal_true():
    """ Test with Number with no precision and valid scale case 1"""
    schema = Schema({"number": Number(scale=2, yield_decimal=True)})
    out_ = schema({"number": '123456789.34'})
    assert float(out_.get("number")) == 123456789.34


def test_number_when_precision_none_n_valid_scale_case2_yield_decimal_true():
    """ Test with Number with no precision and valid scale case 2 with zero in decimal part"""
    schema = Schema({"number": Number(scale=2, yield_decimal=True)})
    out_ = schema({"number": '123456789012.00'})
    assert float(out_.get("number")) == 123456789012.00


def test_number_when_precision_none_n_invalid_scale_yield_decimal_true():
    """ Test with Number with no precision and invalid scale"""
    schema = Schema({"number": Number(scale=2, yield_decimal=True)})
    with pytest.raises(MultipleInvalid) as e:
        schema({"number": '12345678901.234'})
    assert str(e.value) == "Scale must be equal to 2 for dictionary value @ data['number']"


def test_number_when_valid_precision_n_scale_none_yield_decimal_true():
    """ Test with Number with no precision and valid scale"""
    schema = Schema({"number": Number(precision=14, yield_decimal=True)})
    out_ = schema({"number": '1234567.8901234'})
    assert float(out_.get("number")) == 1234567.8901234


def test_number_when_invalid_precision_n_scale_none_yield_decimal_true():
    """ Test with Number with no precision and invalid scale"""
    schema = Schema({"number": Number(precision=14, yield_decimal=True)})
    with pytest.raises(MultipleInvalid) as e:
        schema({"number": '12345674.8901234'})
    assert str(e.value) == "Precision must be equal to 14 for dictionary value @ data['number']"


def test_number_validation_with_valid_precision_scale_yield_decimal_false():
    """ Test with Number with valid precision, scale and no yield_decimal"""
    schema = Schema({"number": Number(precision=6, scale=2, yield_decimal=False)})
    out_ = schema({"number": '1234.00'})
    assert out_.get("number") == '1234.00'


def test_named_tuples_validate_as_tuples():
    NT = collections.namedtuple('NT', ['a', 'b'])
    nt = NT(1, 2)
    t = (1, 2)

    Schema((int, int))(nt)
    Schema((int, int))(t)
    Schema(NT(int, int))(nt)
    Schema(NT(int, int))(t)


def test_datetime():
    schema = Schema({"datetime": Datetime()})
    schema({"datetime": "2016-10-24T14:01:57.102152Z"})
    with pytest.raises(MultipleInvalid):
        schema({"datetime": "2016-10-24T14:01:57"})


def test_date():
    schema = Schema({"date": Date()})
    schema({"date": "2016-10-24"})
    with pytest.raises(MultipleInvalid):
        schema({"date": "2016-10-24Z"})


def test_date_custom_format():
    schema = Schema({"date": Date("%Y%m%d")})
    schema({"date": "20161024"})
    with pytest.raises(MultipleInvalid):
        schema({"date": "2016-10-24"})


def test_ordered_dict():
    if not hasattr(collections, 'OrderedDict'):
        # collections.OrderedDict was added in Python2.7; only run if present
        return
    schema = Schema({Number(): Number()})  # x, y pairs (for interpolation or something)
    data = collections.OrderedDict([(5.0, 3.7), (24.0, 8.7), (43.0, 1.5),
                                    (62.0, 2.1), (71.5, 6.7), (90.5, 4.1),
                                    (109.0, 3.9)])
    out = schema(data)
    assert isinstance(out, collections.OrderedDict), 'Collection is no longer ordered'
    assert data.keys() == out.keys(), 'Order is not consistent'


def test_marker_hashable():
    """Verify that you can get schema keys, even if markers were used"""
    definition = {
        Required('x'): int, Optional('y'): float,
        Remove('j'): int, Remove(int): str, int: int
    }
    assert definition.get('x') == int
    assert definition.get('y') == float
    assert Required('x') == Required('x')
    assert Required('x') != Required('y')
    # Remove markers are not hashable
    assert definition.get('j') == None


def test_schema_infer():
    schema = Schema.infer({
        'str': 'foo',
        'bool': True,
        'int': 42,
        'float': 3.14
    })
    assert schema == Schema({
        Required('str'): str,
        Required('bool'): bool,
        Required('int'): int,
        Required('float'): float
    })


def test_schema_infer_dict():
    schema = Schema.infer({
        'a': {
            'b': {
                'c': 'foo'
            }
        }
    })

    assert schema == Schema({
        Required('a'): {
            Required('b'): {
                Required('c'): str
            }
        }
    })


def test_schema_infer_list():
    schema = Schema.infer({
        'list': ['foo', True, 42, 3.14]
    })

    assert schema == Schema({
        Required('list'): [str, bool, int, float]
    })


def test_schema_infer_scalar():
    assert Schema.infer('foo') == Schema(str)
    assert Schema.infer(True) == Schema(bool)
    assert Schema.infer(42) == Schema(int)
    assert Schema.infer(3.14) == Schema(float)
    assert Schema.infer({}) == Schema(dict)
    assert Schema.infer([]) == Schema(list)


def test_schema_infer_accepts_kwargs():
    schema = Schema.infer({
        'str': 'foo',
        'bool': True
    }, required=False, extra=True)

    # Subset of schema should be acceptable thanks to required=False.
    schema({'bool': False})

    # Keys that are in schema should still match required types.
    with pytest.raises(Invalid):
        schema({'str': 42})

    # Extra fields should be acceptable thanks to extra=True.
    schema({'str': 'bar', 'int': 42})


def test_validation_performance():
    """
    This test comes to make sure the validation complexity of dictionaries is done in a linear time.
    To achieve this a custom marker is used in the scheme that counts each time it is evaluated.
    By doing so we can determine if the validation is done in linear complexity.
    Prior to issue https://github.com/alecthomas/voluptuous/issues/259 this was exponential.
    """
    num_of_keys = 1000

    schema_dict = {}
    data = {}
    data_extra_keys = {}

    counter = [0]

    class CounterMarker(Marker):
        def __call__(self, *args, **kwargs):
            counter[0] += 1
            return super(CounterMarker, self).__call__(*args, **kwargs)

    for i in range(num_of_keys):
        schema_dict[CounterMarker(str(i))] = str
        data[str(i)] = str(i)
        data_extra_keys[str(i * 2)] = str(i)  # half of the keys are present, and half aren't

    schema = Schema(schema_dict, extra=ALLOW_EXTRA)

    schema(data)

    assert counter[0] <= num_of_keys, "Validation complexity is not linear! %s > %s" % (counter[0], num_of_keys)

    counter[0] = 0  # reset counter
    schema(data_extra_keys)

    assert counter[0] <= num_of_keys, "Validation complexity is not linear! %s > %s" % (counter[0], num_of_keys)


def test_IsDir():
    schema = Schema(IsDir())
    with pytest.raises(MultipleInvalid):
        schema(3)
    schema(os.path.dirname(os.path.abspath(__file__)))


def test_IsFile():
    schema = Schema(IsFile())
    with pytest.raises(MultipleInvalid):
        schema(3)
    schema(os.path.abspath(__file__))


def test_PathExists():
    schema = Schema(PathExists())
    with pytest.raises(MultipleInvalid):
        schema(3)
    schema(os.path.abspath(__file__))


def test_description():
    marker = Marker(Schema(str), description='Hello')
    assert marker.description == 'Hello'

    optional = Optional('key', description='Hello')
    assert optional.description == 'Hello'

    exclusive = Exclusive('alpha', 'angles', description='Hello')
    assert exclusive.description == 'Hello'

    inclusive = Inclusive('alpha', 'angles', description='Hello')
    assert inclusive.description == 'Hello'

    required = Required('key', description='Hello')
    assert required.description == 'Hello'


def test_SomeOf_min_validation():
    validator = All(Length(min=8), SomeOf(
        min_valid=3,
        validators=[Match(r'.*[A-Z]', 'no uppercase letters'),
                    Match(r'.*[a-z]', 'no lowercase letters'),
                    Match(r'.*[0-9]', 'no numbers'),
                    Match(r'.*[$@$!%*#?&^:;/<,>|{}()\-\'._+=]', 'no symbols')]))

    validator('ffe532A1!')
    with raises(MultipleInvalid, 'length of value must be at least 8'):
        validator('a')

    with raises(MultipleInvalid, 'no uppercase letters, no lowercase letters'):
        validator('wqs2!#s111')

    with raises(MultipleInvalid, 'no lowercase letters, no symbols'):
        validator('3A34SDEF5')


def test_SomeOf_max_validation():
    validator = SomeOf(
        max_valid=2,
        validators=[Match(r'.*[A-Z]', 'no uppercase letters'),
                    Match(r'.*[a-z]', 'no lowercase letters'),
                    Match(r'.*[0-9]', 'no numbers')],
        msg='max validation test failed')

    validator('Aa')
    with raises(TooManyValid, 'max validation test failed'):
        validator('Aa1')


def test_self_validation():
    schema = Schema({"number": int,
                     "follow": Self})
    with pytest.raises(MultipleInvalid):
        schema({"number": "abc"})
    with pytest.raises(MultipleInvalid):
        schema({"follow": {"number": '123456.712'}})
    schema({"follow": {"number": 123456}})
    schema({"follow": {"follow": {"number": 123456}}})


def test_any_error_has_path():
    """https://github.com/alecthomas/voluptuous/issues/347"""
    s = Schema({
        Optional('q'): int,
        Required('q2'): Any(int, msg='toto')
    })
    with pytest.raises(MultipleInvalid) as e:
        s({'q': 'str', 'q2': 'tata'})
    assert (
        (e.value.errors[0].path == ['q'] and e.value.errors[1].path == ['q2']) or
        (e.value.errors[1].path == ['q'] and e.value.errors[0].path == ['q2'])
    )


def test_all_error_has_path():
    """https://github.com/alecthomas/voluptuous/issues/347"""
    s = Schema({
        Optional('q'): int,
        Required('q2'): All([str, Length(min=10)], msg='toto'),
    })
    with pytest.raises(MultipleInvalid) as e:
        s({'q': 'str', 'q2': 12})
    assert (
        (e.value.errors[0].path == ['q'] and e.value.errors[1].path == ['q2']) or
        (e.value.errors[1].path == ['q'] and e.value.errors[0].path == ['q2'])
    )


def test_match_error_has_path():
    """https://github.com/alecthomas/voluptuous/issues/347"""
    s = Schema({
        Required('q2'): Match("a"),
    })
    with pytest.raises(MultipleInvalid) as e:
        s({'q2': 12})
    assert e.value.errors[0].path == ['q2']


def test_self_any():
    schema = Schema({"number": int,
                     "follow": Any(Self, "stop")})
    with pytest.raises(MultipleInvalid):
        schema({"number": "abc"})
    with pytest.raises(MultipleInvalid):
        schema({"follow": {"number": '123456.712'}})
    schema({"follow": {"number": 123456}})
    schema({"follow": {"follow": {"number": 123456}}})
    schema({"follow": {"follow": {"number": 123456, "follow": "stop"}}})


def test_self_all():
    schema = Schema({"number": int,
                     "follow": All(Self,
                                   Schema({"extra_number": int},
                                          extra=ALLOW_EXTRA))},
                    extra=ALLOW_EXTRA)
    with pytest.raises(MultipleInvalid):
        schema({"number": "abc"})
    with pytest.raises(MultipleInvalid):
        schema({"follow": {"number": '123456.712'}})
    schema({"follow": {"number": 123456}})
    schema({"follow": {"follow": {"number": 123456}}})
    schema({"follow": {"number": 123456, "extra_number": 123}})
    with pytest.raises(MultipleInvalid):
        schema({"follow": {"number": 123456, "extra_number": "123"}})


def test_SomeOf_on_bounds_assertion():
    with raises(AssertionError, 'when using "SomeOf" you should specify at least one of min_valid and max_valid'):
        SomeOf(validators=[])


def test_comparing_voluptuous_object_to_str():
    assert Optional('Classification') < 'Name'


def test_set_of_integers():
    schema = Schema({int})
    with raises(Invalid, 'expected a set'):
        schema(42)
    with raises(Invalid, 'expected a set'):
        schema(frozenset([42]))

    schema(set())
    schema(set([42]))
    schema(set([42, 43, 44]))
    with pytest.raises(MultipleInvalid) as e:
        schema(set(['abc']))
    assert str(e.value) == "invalid value in set"


def test_frozenset_of_integers():
    schema = Schema(frozenset([int]))
    with raises(Invalid, 'expected a frozenset'):
        schema(42)
    with raises(Invalid, 'expected a frozenset'):
        schema(set([42]))

    schema(frozenset())
    schema(frozenset([42]))
    schema(frozenset([42, 43, 44]))
    with pytest.raises(MultipleInvalid) as e:
        schema(frozenset(['abc']))
    assert str(e.value) == "invalid value in frozenset"


def test_set_of_integers_and_strings():
    schema = Schema({int, str})
    with raises(Invalid, 'expected a set'):
        schema(42)

    schema(set())
    schema(set([42]))
    schema(set(['abc']))
    schema(set([42, 'abc']))
    with pytest.raises(MultipleInvalid) as e:
        schema(set([None]))
    assert str(e.value) == "invalid value in set"


def test_frozenset_of_integers_and_strings():
    schema = Schema(frozenset([int, str]))
    with raises(Invalid, 'expected a frozenset'):
        schema(42)

    schema(frozenset())
    schema(frozenset([42]))
    schema(frozenset(['abc']))
    schema(frozenset([42, 'abc']))
    with pytest.raises(MultipleInvalid) as e:
        schema(frozenset([None]))
    assert str(e.value) == "invalid value in frozenset"


def test_lower_util_handles_various_inputs():
    assert Lower(3) == "3"
    assert Lower(u"3") == u"3"
    assert Lower(b'\xe2\x98\x83'.decode("UTF-8")) == b'\xe2\x98\x83'.decode("UTF-8")
    assert Lower(u"A") == u"a"


def test_upper_util_handles_various_inputs():
    assert Upper(3) == "3"
    assert Upper(u"3") == u"3"
    assert Upper(b'\xe2\x98\x83'.decode("UTF-8")) == b'\xe2\x98\x83'.decode("UTF-8")
    assert Upper(u"a") == u"A"


def test_capitalize_util_handles_various_inputs():
    assert Capitalize(3) == "3"
    assert Capitalize(u"3") == u"3"
    assert Capitalize(b'\xe2\x98\x83'.decode("UTF-8")) == b'\xe2\x98\x83'.decode("UTF-8")
    assert Capitalize(u"aaa aaa") == u"Aaa aaa"


def test_title_util_handles_various_inputs():
    assert Title(3) == "3"
    assert Title(u"3") == u"3"
    assert Title(b'\xe2\x98\x83'.decode("UTF-8")) == b'\xe2\x98\x83'.decode("UTF-8")
    assert Title(u"aaa aaa") == u"Aaa Aaa"


def test_strip_util_handles_various_inputs():
    assert Strip(3) == "3"
    assert Strip(u"3") == u"3"
    assert Strip(b'\xe2\x98\x83'.decode("UTF-8")) == b'\xe2\x98\x83'.decode("UTF-8")
    assert Strip(u" aaa ") == u"aaa"


def test_any_required():
    schema = Schema(Any({'a': int}, {'b': str}, required=True))

    with pytest.raises(MultipleInvalid) as e:
        schema({})
    assert str(e.value) == "required key not provided @ data['a']"


def test_any_required_with_subschema():
    schema = Schema(Any({'a': Any(float, int)},
                        {'b': int},
                        {'c': {'aa': int}},
                    required=True))

    with pytest.raises(MultipleInvalid) as e:
        schema({})
    assert str(e.value) == "required key not provided @ data['a']"


def test_inclusive():
    schema = Schema({
        Inclusive('x', 'stuff'): int,
        Inclusive('y', 'stuff'): int,
        })

    r = schema({})
    assert r == {}

    r = schema({'x': 1, 'y': 2})
    assert r == {'x': 1, 'y': 2}

    with pytest.raises(MultipleInvalid) as e:
        r = schema({'x': 1})
    assert str(e.value) == "some but not all values in the same group of inclusion 'stuff' @ data[<stuff>]"


def test_inclusive_defaults():
    schema = Schema({
        Inclusive('x', 'stuff', default=3): int,
        Inclusive('y', 'stuff', default=4): int,
        })

    r = schema({})
    assert r == {'x': 3, 'y': 4}

    with pytest.raises(MultipleInvalid) as e:
        r = schema({'x': 1})
    assert str(e.value) == "some but not all values in the same group of inclusion 'stuff' @ data[<stuff>]"


def test_exclusive():
    schema = Schema({
        Exclusive('x', 'stuff'): int,
        Exclusive('y', 'stuff'): int,
        })

    r = schema({})
    assert r == {}

    r = schema({'x': 1})
    assert r == {'x': 1}

    with pytest.raises(MultipleInvalid) as e:
        r = schema({'x': 1, 'y': 2})
    assert str(e.value) == "two or more values in the same group of exclusion 'stuff' @ data[<stuff>]"


def test_any_with_discriminant():
    schema = Schema({
        'implementation': Union({
            'type': 'A',
            'a-value': str,
        }, {
            'type': 'B',
            'b-value': int,
        }, {
            'type': 'C',
            'c-value': bool,
        }, discriminant=lambda value, alternatives: filter(lambda v: v['type'] == value['type'], alternatives))
    })
    with pytest.raises(MultipleInvalid) as e:
        schema({
            'implementation': {
                'type': 'C',
                'c-value': None
            }
        })
    assert str(e.value) == 'expected bool for dictionary value @ data[\'implementation\'][\'c-value\']'

def test_coerce_enum():
    """Test Coerce Enum"""
    class Choice(Enum):
        Easy = 1
        Medium = 2
        Hard = 3

    class StringChoice(str, Enum):
        Easy = "easy"
        Medium = "medium"
        Hard = "hard"

    schema = Schema(Coerce(Choice))
    string_schema = Schema(Coerce(StringChoice))

    # Valid value
    assert schema(1) == Choice.Easy
    assert string_schema("easy") == StringChoice.Easy

    # Invalid value
    with pytest.raises(Invalid) as e:
        schema(4)
    assert str(e.value) == "expected Choice or one of 1, 2, 3"

    with pytest.raises(Invalid) as e:
        string_schema("hello")
    assert str(e.value) == "expected StringChoice or one of 'easy', 'medium', 'hard'"
