class Composable:

    def __init__(self):
        super().__init__()

    def then(self, f, *args, **kw):
        """Compose this processor with a new processor defined by a function.

        The function is of the form:

            def my_process(source, ...):
                for sample in source:
                    ...
                    result = ...
                    yield result
        """
        assert callable(f)
        assert "source" not in kw
        return Processor(self, f, *args, **kw)

class Shorthands:
    pass 
    
    
class Processor(Composable, Shorthands):
    """A class that turns a function into an Iterable."""

    def __init__(self, source, f, *args, **kwargs):
        
        super().__init__()
        assert callable(f)
        self.source = source
        self.f = f
        self.args = args
        self.kwargs = kwargs

    def __iter__(self):
        """Return an iterator over the source dataset processed by the given function."""
        assert self.source is not None, f"must set source before calling iter {self.f} {self.args} {self.kwargs}"
        assert callable(self.f), self.f
        return self.f(iter(self.source), *self.args, **self.kwargs)
