class Foo:
    def func_a(self):
        # should be counted
        pass


def func_b():
    # should be counted
    pass


def func_c():
    # should be counted
    def func_d():
        # should not be counted
        pass
