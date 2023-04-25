from typing import Any
import threading


class ModelsBase(object):
    def __init__(self):
        self.models = {}
        self.lock = threading.Lock()

    def unload(self):
        with self.lock:
            for name, model in self.models.items():
                delattr(self, name)
                del model

    def __getitem__(self, item):
        with self.lock:
            if hasattr(self, item):
                return getattr(self, item)

            self.load(item)

            return getattr(self, item)

    def load(self, item: str) -> None:
        raise NotImplementedError

    def register(self, name: str, model: Any) -> None:
        with self.lock:
            if name in self.models:
                del self.models[name]
                delattr(self, name)
            self.models[name] = model
            setattr(self, name, model)
