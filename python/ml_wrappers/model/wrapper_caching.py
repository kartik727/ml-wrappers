# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines a decorator for wrapper models that caches the prediction."""

import json


class CachedModel:
    """A wrapper that caches predictions in memory and optionally to disk."""

    def __init__(self, model, cache_filename=None):
        self.model = model
        self.cache_filename = cache_filename
        self.cache = {}

    def load_cache(self):
        """Load the cache from the cache file."""
        if self.cache_filename is None:
            print('No cache filename provided, skipping load_cache')
            return

        try:
            with open(self.cache_filename, 'r') as f:
                self.cache = json.load(f)
        except FileNotFoundError:
            print(f'cache file "{self.cache_filename}" not found, skipping load_cache')

    def save_cache(self):
        """Write the cache to the cache file."""
        if self.cache_filename is None:
            print('No cache filename provided, skipping save_cache')
            return

        with open(self.cache_filename, 'w') as f:
            json.dump(self.cache, f, indent=4)

    def predict(self, context, model_input=None):
        """Predict the output using the wrapped model."""
        if model_input is None:
            model_input = context

        try:
            key = hash(model_input)
        except TypeError:
            key = None
        if key and key in self.cache:
            print('Cache hit')
            return self.cache[key]
        print('Cache miss')
        response = self.model.predict(context, model_input)
        self.cache[key] = response
        return response


class CachedDecorator:
    """A decorator for the CachedModel class."""
    def __init__(self, model_cls):
        self.model_cls = model_cls

    def __call__(self, *args, cache_filename=None, **kwargs):
        return CachedModel(self.model_cls(*args, **kwargs), cache_filename)
