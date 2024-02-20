from tofina.constants import ASSET_CACHE_KEY, INSTRUMENT_CACHE_KEY


class CalculationCache:
    def __init__(self):
        self.storage = {}
        self.allowed_keys = set()

    def register_key(self, key):
        self.allowed_keys.add(key)

    def update(self, key, value):
        self.storage[key] = value

    def use_stale_assets_and_instruments(self):
        cache_assets = ASSET_CACHE_KEY in self.allowed_keys
        cache_instruments = INSTRUMENT_CACHE_KEY in self.allowed_keys
        return cache_assets and cache_instruments

    def invalidate_all_cache(self):
        self.storage = {}

    def __call__(self, f, key):
        def wrapper(*args, **kwargs):
            if key not in self.allowed_keys:
                return f(*args, **kwargs)
            if key not in self.storage:
                self.storage[key] = f(*args, **kwargs)
            return self.storage[key]

        return wrapper
