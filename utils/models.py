from typing import Any
import threading
from abc import ABC, abstractmethod


class ModelsBase(ABC):
    def __init__(self):
        self.models = {}
        self.lock = threading.Lock()

    def unload(self):
        with self.lock:
            for name, model in self.models.items():
                delattr(self, name)
                del model

    def __getattr__(self, item):
        print(f'Getting {item} ...')
        if hasattr(self, item):
            return getattr(self, item)

        self.load(item)

        return getattr(self, item)

    def load(self, item: str) -> None:
        raise NotImplementedError

    def register(self, name: str, model: Any) -> None:
        print(f'Register {name} ... {getattr(self, name)}')
        with self.lock:
            if name in self.models:
                print(f"Unloading {name} ...")
                delattr(self, name)
                del self.models[name]
            print(f"Loading {name} ...")
            self.models[name] = model

            setattr(self, name, model)
