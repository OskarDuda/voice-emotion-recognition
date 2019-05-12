from emotions import lib


def test_fun_call_by_name():
    assert callable(lib.fun_call_by_name('count_data'))
    assert callable(lib.fun_call_by_name('scipy.average'))
    assert lib.fun_call_by_name('scipy.average')([1, 2, 3]) == 2
