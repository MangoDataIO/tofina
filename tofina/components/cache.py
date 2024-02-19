class CalculationCache:
    def __init__(self):
        self.storage = {}
        self.allowed_keys = set()

    def register_key(self, key):
        self.allowed_keys.add(key)

    def __call__(self, f, key):
        def wrapper(*args, **kwargs):
            if key not in self.allowed_keys:
                return f(*args, **kwargs)
            if key not in self.storage:
                self.storage[key] = f(*args, **kwargs)
            return self.storage[key]

        return wrapper
