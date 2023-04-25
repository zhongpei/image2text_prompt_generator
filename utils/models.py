from typing import Any
import threading
from abc import ABC, abstractmethod


class ModelsBase(object):
    def __init__(self):
        self.models = {}
        self.lock = threading.Lock()

    def unload(self):
        with self.lock:
            for _, model in self.models.items():
                del model

    def __getattr__(self, item):

        print(f'Getting {item} ...')

        if item in self.models:
            return self.models[item]

        self.load(item)
        if item in self.models:
            return self.models[item]

        return getattr(self, item)

    def load(self, item: str) -> None:
        raise NotImplementedError

    def register(self, name: str, model: Any) -> None:
        print(f'Register {name} ...')

        if name in self.models:
            print(f"Unloading {name} ...")
            # delattr(self, name)
            del self.models[name]

        self.models[name] = model

        # setattr(self, name, model)
