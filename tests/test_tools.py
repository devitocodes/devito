from concurrent.futures import ThreadPoolExecutor
from threading import RLock
import numpy as np
import pytest
from sympy.abc import a, b, c, d, e

import time

from devito import Operator, Eq
from devito.tools import (UnboundedMultiTuple, ctypes_to_cstr, toposort,
                          filter_ordered, transitive_closure, UnboundTuple,
                          CacheInstances, memoized_meth, memoized_generator)
from devito.types.basic import Symbol


@pytest.mark.parametrize('elements, expected', [
    ([[a, b, c], [c, d, e]], [a, b, c, d, e]),
    ([[e, d, c], [c, b, a]], [e, d, c, b, a]),
    ([[a, b, c], [b, d, e]], [a, b, d, c, e]),
    ([[a, b, c], [d, b, c]], [a, d, b, c]),
    ([[a, b, c], [c, d, b]], None),
])
def test_toposort(elements, expected):
    try:
        ordering = toposort(elements)
        assert ordering == expected
    except ValueError:
        assert expected is None


def test_sorting():
    key = lambda x: x

    # Need predictable random sequence or test will
    # have inconsistent behaviour results between tests.
    np.random.seed(0)
    array = np.random.randint(-1000, 1000, 10000)

    t0 = time.time()
    for _ in range(100):
        sort_key = filter_ordered(array, key=key)
    t1 = time.time()
    for _ in range(100):
        sort_nokey = filter_ordered(array)
    t2 = time.time()

    assert t2 - t1 < 0.8 * (t1 - t0)
    assert sort_key == sort_nokey


def test_transitive_closure():
    a = Symbol('a')
    b = Symbol('b')
    c = Symbol('c')
    d = Symbol('d')
    e = Symbol('e')
    f = Symbol('f')

    mapper = {a: b, b: c, c: d, f: e}
    mapper = transitive_closure(mapper)
    assert mapper == {a: d, b: d, c: d, f: e}


def test_loops_in_transitive_closure():
    a = Symbol('a')
    b = Symbol('b')
    c = Symbol('c')
    d = Symbol('d')
    e = Symbol('e')

    mapper = {a: b, b: c, c: d, d: e, e: b}
    mapper = transitive_closure(mapper)
    assert mapper == {a: b, b: c, c: d, d: e, e: b}


@pytest.mark.parametrize('mapper, expected', [
    ([{a: b, b: a, c: d, d: e, e: c}, [a, a, c, c, c]]),
    ([{a: b, b: c, c: b, d: e, e: d}, [b, b, b, d, d]]),
    ([{a: c, b: a, c: a, d: e, e: d}, [a, a, a, d, d]]),
    ([{c: a, b: a, a: c, e: c, d: e}, [a, a, a, c, c]]),
    ([{a: b, b: c, c: d, d: e, e: b}, [b, b, b, b, b]]),
])
def test_sympy_subs_symmetric(mapper, expected):
    a = Symbol('a')
    b = Symbol('b')
    c = Symbol('c')
    d = Symbol('d')
    e = Symbol('e')

    input = [a, b, c, d, e]
    input = [i.subs(mapper) for i in input]
    assert input == expected


@pytest.mark.parametrize('dtype, expected', [
    (np.float32, 'float'),
    (np.float64, 'double'),
    (np.int32, 'int'),
    (np.int64, 'long'),
    (np.uint64, 'unsigned long'),
    (np.int8, 'char'),
    (np.uint8, 'unsigned char'),
])
def test_ctypes_to_cstr(dtype, expected):
    a = Symbol(name='a', dtype=dtype)
    assert ctypes_to_cstr(a._C_ctype) == expected


def test_unbounded_multi_tuple():
    ub = UnboundedMultiTuple([1, 2], [3, 4])
    with pytest.raises(StopIteration):
        ub.next()

    with pytest.raises(StopIteration):
        assert ub.curitem()

    ub.iter()
    assert ub.curitem() == (1, 2)
    assert ub.next() == 1
    assert ub.next() == 2

    with pytest.raises(StopIteration):
        ub.next()

    ub.iter()
    assert ub.next() == 3
    assert ub.next() == 4

    with pytest.raises(StopIteration):
        ub.next()

    ub.iter()
    assert ub.next() == 3

    assert ub.nextitem() == (3, 4)


def test_unbound_tuple():
    # Make sure we don't drop needed None for 2.5d
    ub = UnboundTuple(None, None)
    assert len(ub) == 2
    assert ub[10] is None

    ub = UnboundTuple(1, 2, 3)
    assert len(ub) == 3
    assert ub[10] == 3
    assert ub[1:4] == (2, 3, 3)
    assert ub.next() == 1
    assert ub.next() == 2
    ub.iter()
    assert ub.next() == 1


class TestCacheInstances:

    def test_caching(self):
        """
        Tests basic functionality of cached instances.
        """
        class Object(CacheInstances):
            def __init__(self, value: int):
                self.value = value

        obj1 = Object(1)
        obj2 = Object(1)
        obj3 = Object(2)

        assert obj1 is obj2
        assert obj1 is not obj3

    def test_cache_size(self):
        """
        Tests specifying the size of the instance cache.
        """
        class Object(CacheInstances):
            _instance_cache_size = 2

            def __init__(self, value: int):
                self.value = value

        obj1 = Object(1)
        obj2 = Object(2)
        obj3 = Object(3)
        obj4 = Object(1)
        obj5 = Object(3)

        # obj1 should have been evicted before obj4 was created
        assert obj1 is not obj4
        assert obj1 is not obj2
        assert obj3 is obj5

        hits, _, _, cursize = Object._instance_cache.cache_info()
        assert hits == 1  # obj5 hit the cache
        assert cursize == 2

    def test_cleared_after_build(self):
        """
        Tests that instance caches are cleared after building an Operator.
        """
        class Object(CacheInstances):
            def __init__(self, value: int):
                self.value = value

        obj1 = Object(1)
        cache_size = Object._instance_cache.cache_info()[-1]
        assert cache_size == 1

        x = Symbol('x')
        Operator(Eq(x, obj1.value))

        # Cache should be cleared after Operator construction
        cache_size = Object._instance_cache.cache_info()[-1]
        assert cache_size == 0


class TestMemoizedMethods:

    def test_memoized_meth(self):
        """
        Tests basic functionality of memoized_meth
        """
        class Object:
            def __init__(self):
                self.misses = 0

            @memoized_meth
            def compute(self, x):
                self.misses += 1
                return x * 2

        obj = Object()
        obj.compute(2)
        obj.compute(4)
        assert obj.compute(2) == 4
        assert obj.compute(4) == 8
        assert obj.misses == 2  # Only two unique calls

    def test_unhashable_args(self):
        """
        Tests that memoized_meth raises an error for unhashable arguments.
        """
        class Object:
            def __init__(self):
                self.misses = 0

            @memoized_meth
            def compute(self, x: list[int]):
                self.misses += 1
                return sum(x)

        obj = Object()
        with pytest.raises(TypeError):
            obj.compute([1, 2, 3])

    @pytest.mark.parametrize('num_threads', [5, 11, 17])
    def test_memoized_meth_concurrency(self, num_threads: int):
        """
        Tests concurrent calls to a memoized method
        """
        # Each thread should have its own cache; the calls should not block
        class Object:
            def __init__(self):
                self.misses = 0
                self.lock = RLock()

            @memoized_meth
            def compute(self, x):
                # print ID of the running thread
                with self.lock:
                    self.misses += 1

                # Simulate some computation
                time.sleep(0.2)
                return x * 2

        obj = Object()
        def worker(x: int) -> int:
            a = obj.compute(x)
            b = obj.compute(x)
            assert a == b
            return a

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            stime = time.perf_counter()
            futures = [executor.submit(worker, i % 4) for i in range(num_threads)]
            results = [f.result() for f in futures]
            etime = time.perf_counter()

        assert len(set(results)) == 4  # Should have gotten four unique results
        assert obj.misses == num_threads  # Each thread should have missed once

        # Ensure that the total time is approximately 0.2 seconds (one miss per thread)
        expected = 0.2
        assert abs(etime - stime - expected) < 0.1 * expected

    def test_memoized_generator(self):
        """
        Tests basic functionality of memoized_generator
        """
        class Object:
            def __init__(self):
                self.misses = 0

            @memoized_generator
            def compute(self, x):
                self.misses += 1
                yield x * 2
                yield x * 3

        obj = Object()
        list(obj.compute(2))
        assert tuple(obj.compute(2)) == (4, 6)
        assert obj.misses == 1  # Only one unique call

    @pytest.mark.parametrize('num_threads', [5, 11, 17])
    def test_memoized_generator_concurrency(self, num_threads: int):
        """
        Tests concurrent calls to a memoized generator
        """
        class Object:
            def __init__(self):
                self.misses = 0
                self.lock = RLock()

            @memoized_generator
            def compute(self, x):
                with self.lock:
                    self.misses += 1

                time.sleep(0.25)
                yield x * 2

                time.sleep(0.25)
                yield x * 3

        # With memoized_generator, the initial construction should block but iteration
        # should be concurrent and reuse the same iterator.

        obj = Object()
        def worker(x: int) -> list[int]:
            return list(obj.compute(x))

        # If one thread consumes the generator, subsequent iteration shouldn't block
        # First we iterate concurrently; all but one thread should block to wait for
        # the producing thread, so all will take ~0.5 seconds
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            stime = time.perf_counter()
            futures = [executor.submit(worker, i % 4) for i in range(num_threads)]
            results = [f.result() for f in futures]
            etime = time.perf_counter()

        expected = 0.5
        assert abs(etime - stime - expected) < 0.1 * expected
        assert set(tuple(r) for r in results) == {(0, 0), (2, 3), (4, 6), (6, 9)}
        assert obj.misses == 4  # One miss per unique call

        # Now iterating the same calls should use buffered generators from the cache
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            stime = time.perf_counter()
            futures = [executor.submit(worker, i % 4) for i in range(num_threads)]
            results = [f.result() for f in futures]
            etime = time.perf_counter()

        assert etime - stime < 0.1  # Should take epsilon time
        assert set(tuple(r) for r in results) == {(0, 0), (2, 3), (4, 6), (6, 9)}
        assert obj.misses == 4  # No new misses; all calls reused cached generators
