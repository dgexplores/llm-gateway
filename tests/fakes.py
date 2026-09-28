"""A provider that can be told to fail, for exercising failover and retry paths."""

from __future__ import annotations

from gateway.llm.base import Completion, Message


class FakeProvider:
    """Returns a scripted sequence of results.

    Each item is either a ``Completion`` to return, or an exception to raise. Once the
    script is exhausted the last item repeats, which is how "always fails" is expressed.
    """

    def __init__(self, name: str, script: list[Completion | Exception], model: str = "fake-1"):
        self.name = name
        self.model = model
        self._script = script
        self.calls = 0

    def complete(self, messages: list[Message], *, temperature: float = 0.0) -> Completion:
        index = min(self.calls, len(self._script) - 1)
        self.calls += 1
        result = self._script[index]
        if isinstance(result, Exception):
            raise result
        return result

    def healthy(self) -> bool:
        return True
