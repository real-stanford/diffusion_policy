#!/usr/bin/python
#
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import inspect
import os

from gym.envs.registration import registry as gym_registry


def import_class_from_path(class_path):
    """Given 'path.to.module:object', imports and returns the object."""
    module_path, class_name = class_path.split(":")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class ConfigCache(object):
    """Configuration class to store constructor arguments.

    This is used to store parameters to pass to Gym environments at init time.
    """

    def __init__(self):
        self._configs = {}
        self._default_config = {}

    def set_default_config(self, config):
        """Sets the default configuration used for all RobotEnv envs."""
        self._default_config = dict(config)

    def set_config(self, cls_or_env_id, config):
        """Sets the configuration for the given environment within a context.

        Args:
            cls_or_env_id (Class | str): A class type or Gym environment ID to
                configure.
            config (dict): The configuration parameters.
        """
        config_key = self._get_config_key(cls_or_env_id)
        self._configs[config_key] = dict(config)

    def get_config(self, cls_or_env_id):
        """Returns the configuration for the given env name.

        Args:
            cls_or_env_id (Class | str): A class type or Gym environment ID to
                get the configuration of.
        """
        config_key = self._get_config_key(cls_or_env_id)
        config = dict(self._default_config)
        config.update(self._configs.get(config_key, {}))
        return config

    def clear_config(self, cls_or_env_id):
        """Clears the configuration for the given ID."""
        config_key = self._get_config_key(cls_or_env_id)
        if config_key in self._configs:
            del self._configs[config_key]

    def _get_config_key(self, cls_or_env_id):
        if inspect.isclass(cls_or_env_id):
            return cls_or_env_id
        env_id = cls_or_env_id
        assert isinstance(env_id, str)
        if env_id not in gym_registry.env_specs:
            raise ValueError("Unregistered environment name {}.".format(env_id))
        entry_point = gym_registry.env_specs[env_id]._entry_point
        if callable(entry_point):
            return entry_point
        else:
            return import_class_from_path(entry_point)


# Global robot config.
global_config = ConfigCache()


def configurable(config_id=None, pickleable=False, config_cache=global_config):
    """Class decorator to allow injection of constructor arguments.

    This allows constructor arguments to be passed via ConfigCache.
    Example usage:

    @configurable()
    class A:
        def __init__(b=None, c=2, d='Wow'):
            ...

    global_config.set_config(A, {'b': 10, 'c': 20})
    a = A()      # b=10, c=20, d='Wow'
    a = A(b=30)  # b=30, c=20, d='Wow'

    Args:
        config_id: ID of the config to use. This defaults to the class type.
        pickleable: Whether this class is pickleable. If true, causes the pickle
            state to include the config and constructor arguments.
        config_cache: The ConfigCache to use to read config data from. Uses
            the global ConfigCache by default.
    """
    def cls_decorator(cls):
        assert inspect.isclass(cls)

        # Overwrite the class constructor to pass arguments from the config.
        base_init = cls.__init__
        def __init__(self, *args, **kwargs):

            config = config_cache.get_config(config_id or type(self))
            # Allow kwargs to override the config.
            kwargs = {**config, **kwargs}

            # print('Initializing {} with params: {}'.format(type(self).__name__,
                                                           # kwargs))

            if pickleable:
                self._pkl_env_args = args
                self._pkl_env_kwargs = kwargs

            base_init(self, *args, **kwargs)
        cls.__init__ = __init__

        # If the class is pickleable, overwrite the state methods to save
        # the constructor arguments and config.
        if pickleable:
            # Use same pickle keys as gym.utils.ezpickle for backwards compat.
            PKL_ARGS_KEY = '_ezpickle_args'
            PKL_KWARGS_KEY = '_ezpickle_kwargs'

            def __getstate__(self):
                return {
                    PKL_ARGS_KEY: self._pkl_env_args,
                    PKL_KWARGS_KEY: self._pkl_env_kwargs,
                }
            cls.__getstate__ = __getstate__

            def __setstate__(self, data):
                saved_args = data[PKL_ARGS_KEY]
                saved_kwargs = data[PKL_KWARGS_KEY]

                # Override the saved state with the current config.
                config = config_cache.get_config(config_id or type(self))
                # Allow kwargs to override the config.
                kwargs = {**saved_kwargs, **config}

                inst = type(self)(*saved_args, **kwargs)
                self.__dict__.update(inst.__dict__)
            cls.__setstate__ = __setstate__

        return cls
    return cls_decorator
