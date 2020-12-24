class CallTree:

    def __init__(self, T_objs, T, *fn_names):
        self.T_objs = T_objs
        for fn_name in fn_names:
            def func(self, *args, **kwargs):
                for T_obj in self.T_objs:
                    T_obj.fn_name(*args, **kwargs)
            self.__setattr__(fn_name, func)


class Freezable(CallTree):

    def __init__(self, freezables=[]):
        super(Freezable, self).__init__(freezables, Freezable, "freeze", "unfreeze")


class Stepable(CallTree):

    def __init__(self, stepables=[]):
        super(Stepable, self).__init__(stepables, Stepable, "step")


class Trainable(CallTree):

    def __init__(self, trainables=[]):
        super(Trainable, self).__init__(trainables, Trainable, "train")


class PermanentName:

    def __init__(self, name):
        self._name = name

    @property
    def get_name(self):
        return self._name