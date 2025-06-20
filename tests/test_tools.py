from concurrent.futures import Future, ThreadPoolExecutor
from threading import Barrier
from weakref import ref
import numpy as np
import pytest
from sympy.abc import a, b, c, d, e

import time

from devito.tools import (UnboundedMultiTuple, ctypes_to_cstr, toposort,
                          filter_ordered, transitive_closure, UnboundTuple)
from devito.tools.abc import WeakValueCache
from devito.tools.memoization import _memoized_instances
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


class CacheObject:
    # Can't use `object` because it's not weak-referencable
    ...


class TestWeakValueCache:
    """
    Tests for the `WeakValueCache` class and thread safety.
    """

    def test_caching(self) -> None:
        """
        Tests that `WeakValueCache` caches and returns the same instance while it exists.
        """
        cache: WeakValueCache[int, CacheObject] = WeakValueCache()

        old_obj = cache.get_or_create(1, CacheObject)
        new_obj = cache.get_or_create(1, CacheObject)
        oth_obj = cache.get_or_create(2, CacheObject)

        # Ensure the same object is returned for the same key (and vice versa)
        assert new_obj is old_obj
        assert oth_obj is not new_obj

    def test_eviction(self) -> None:
        """
        Tests that `WeakValueCache` does not keep objects alive, and that entries
        are evicted when their values are no longer referenced.
        """
        cache: WeakValueCache[int, CacheObject] = WeakValueCache()
        old_id: int

        # Cache an object and let it immediately drop, storing the memory address
        def scope() -> None:
            nonlocal old_id
            old_obj = cache.get_or_create(1, CacheObject)
            old_id = id(old_obj)

        # Ensure the object is evicted after leaving the scope
        scope()
        new_obj = cache.get_or_create(1, CacheObject)
        assert id(new_obj) != old_id

    def test_clear(self) -> None:
        """
        Tests clearing the cache while objects may still exist.
        """
        cache: WeakValueCache[int, CacheObject] = WeakValueCache()

        # Create an object that stays alive, as well as one that's dropped right away
        obj = cache.get_or_create(1, CacheObject)
        cache.get_or_create(2, CacheObject)

        # Ensure both are dropped from the cache
        assert len(cache) == 1
        cache.clear()
        assert len(cache) == 0

        # Ensure the object is still alive
        assert isinstance(obj, CacheObject)

    def test_safety(self) -> None:
        """
        Tests that `WeakValueCache` is safe for concurrent access with the same key.
        """
        cache: WeakValueCache[int, CacheObject] = WeakValueCache()

        num_threads = 16
        barrier = Barrier(num_threads)

        def worker(_: int) -> CacheObject:
            # Wait until all threads can try to access the cache at once
            barrier.wait()
            return cache.get_or_create(1, CacheObject)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            results = list(executor.map(worker, range(num_threads)))

        # All threads should get the same instance
        ids = {id(obj) for obj in results}
        assert len(ids) == 1
        assert isinstance(results[0], CacheObject)

    def test_concurrent_construction(self) -> None:
        """
        Tests that `WeakValueCache` allows for construction of objects in parallel
        for distinct keys, ensuring each key gets a unique instance.
        """
        cache: WeakValueCache[int, CacheObject] = WeakValueCache()

        num_threads = 24
        num_keys = 6  # Number of unique keys to use
        creation_time = 0.5  # Time to take for constructing a unique object
        expected_time = creation_time * 2  # Max time to expect for all threads to finish

        def supplier() -> CacheObject:
            # Simulate a time-consuming object construction
            time.sleep(creation_time)
            return CacheObject()

        def worker(key: int) -> CacheObject:
            return cache.get_or_create(key, supplier)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            start_time = time.perf_counter()
            results = list(executor.map(worker, (i % num_keys
                                                 for i in range(num_threads))))
            duration = time.perf_counter() - start_time

        # Ensure construction took a reasonable amount of time
        assert duration < expected_time

        # Ensure we constructed unique objects for each key
        ids = {id(obj) for obj in results}
        assert len(ids) == num_keys, "Expected unique objects for each key"

    def test_retry_on_dead_ref(self) -> None:
        """
        Tests that `WeakValueCache` recovers from a race condition where an object
        being constructed is collected before a waiting thread can access it.
        """
        cache: WeakValueCache[int, CacheObject] = WeakValueCache()

        # Create a future that resolves to a dead reference
        dead_ref_future = Future()
        dead_ref_future.set_result(ref(CacheObject()))  # ref is immediately dropped

        # Manually insert the dead reference into the cache
        cache._futures[1] = dead_ref_future

        # Query from another thread while evicting the dead reference on this thread
        def query() -> CacheObject:
            # Should spin until the dead reference is evicted, then populate the cache
            return cache.get_or_create(1, CacheObject)

        with ThreadPoolExecutor(max_workers=1) as executor:
            # Spin up the query thread
            query_future = executor.submit(query)

            # Evict the dead reference
            assert cache._futures[1] is dead_ref_future
            del cache._futures[1]

            # Query should succeed after eviction
            result = query_future.result(timeout=1)
            assert isinstance(result, CacheObject)
            assert len(cache) == 1

    def test_supplier_exception(self):
        """
        Tests that an exception in the supplier is propagated to all waiting threads.
        """
        cache: WeakValueCache[int, CacheObject] = WeakValueCache()

        class SupplierException(Exception):
            pass

        def supplier() -> CacheObject:
            raise SupplierException("Supplier failed")

        num_threads = 16
        exceptions = [None] * num_threads

        def worker(index: int):
            with pytest.raises(SupplierException) as exc_info:
                cache.get_or_create(index, supplier)
            exceptions[index] = exc_info.value

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            executor.map(worker, range(num_threads))

        # Ensure all threads received the exception
        for exc in exceptions:
            assert isinstance(exc, SupplierException)


class TestMemoizedInstances:
    """
    Tests for the `memoized_instances` decorator.
    """

    def test_memo(self):
        """
        Tests basic functionality of memoized instances.
        """
        @_memoized_instances
        class Box:
            def __init__(self, value: int):
                self.value = value
                self.init_calls = getattr(self, 'init_calls', 0) + 1

        # Create instances with the same value
        box1 = Box(10)
        box2 = Box(10)
        box3 = Box(20)

        # Ensure they are the same instance
        assert box1.value == 10
        assert box1 is box2

        # Ensure initialization only happened once
        assert box1.init_calls == 1

        # Ensure different values create different instances
        assert box1 is not box3
        assert box3.init_calls == 1

    def test_memo_with_new(self):
        """
        Tests that `memoized_instances` works correctly with `__new__`.
        """
        @_memoized_instances
        class BoxWithNew:
            def __new__(cls, value: int):
                instance = super().__new__(cls)
                instance.value = value
                return instance

            def __init__(self, value: int):
                self.value += value
                self.init_calls = getattr(self, 'init_calls', 0) + 1

        # Create instances with the same value
        box1 = BoxWithNew(10)
        box2 = BoxWithNew(10)
        box3 = BoxWithNew(20)

        # Ensure they are the same instance
        assert box1.value == 20
        assert box1 is box2

        # Ensure initialization only happened once
        assert box1.init_calls == 1

        # Ensure different values create different instances
        assert box1 is not box3
        assert box3.init_calls == 1

    def test_idempotency(self):
        """
        Tests that applying the decorator multiple times in an inheritance chain
        does not change the behavior.
        """
        @_memoized_instances
        @_memoized_instances
        class Box:
            def __init__(self, value: int):
                self.value = value
                self.init_calls = getattr(self, 'init_calls', 0) + 1

        @_memoized_instances
        class SubBox(Box):
            def __init__(self, value: int):
                super().__init__(value)
                self.sub_init_calls = getattr(self, 'sub_init_calls', 0) + 1

        # Create instances with the same value
        box = Box(10)
        subbox1 = SubBox(10)
        subbox2 = SubBox(10)
        subbox3 = SubBox(20)

        # Ensure the subclass instances are not the same as the base class
        assert box is not subbox1
        assert subbox1 is subbox2
        assert subbox1 is not subbox3

    def test_constructed_elsewhere(self):
        """
        Tests that instances somehow constructed without the replaced new function
        are still initialized correctly (edge case).
        """
        class Box:
            def __new__(cls, _: int):
                return super().__new__(cls)

            def __init__(self, value: int):
                self.value = value
                self.init_calls = getattr(self, 'init_calls', 0) + 1

        # Store the original __new__ method
        original_new = Box.__new__
        Box = _memoized_instances(Box)

        # Create a cached instance as normal
        box1 = Box(10)

        # Restore the original __new__ method and construct a new instance
        Box.__new__ = original_new
        box2 = Box(10)

        # Ensure the new instance is initialized correctly
        assert box2.value == 10
        assert box2.init_calls == 1

        # Ensure the instances are not the same
        assert box1 is not box2
