#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
    Dataset configuration via OmegaConf and Hydra for gymu WebDatasets.

    OmegaConf : https://github.com/omry/omegaconf
    Hydra : https://github.com/facebookresearch/hydra

    Created on 01-06-2022
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"


from typing import Any, Union, Callable, List, Dict

from omegaconf import DictConfig, ListConfig, OmegaConf
import yaml
import gym
import hydra
from functools import partial, reduce
import ast
from types import SimpleNamespace

__all__ = ("transform", 'environment')

def wrapper_resolver(wrapper):
    mod = ast.parse(wrapper.strip())
    assert len(mod.body) == 1 # too many expressions...
    result = _ast_expr_resolver(mod.body[0])
    obj = dict(_target_ = _fun_qual_name(hydra.utils.get_class), _args_ = [result.name])
    return  DictConfig(dict(_target_ = _fun_qual_name(transform), 
                        _args_ = [obj] + result.args, 
                        **result.kwargs))

def transform_resolver(tran):
    mod = ast.parse(tran.strip())
    assert len(mod.body) == 1 # too many expressions...
    result = _ast_expr_resolver(mod.body[0])
    obj = dict(_target_ = _fun_qual_name(hydra.utils.get_method), _args_ = [result.name])
    return  DictConfig(dict(_target_ = _fun_qual_name(transform), 
                        _args_ = [obj] + result.args, 
                        **result.kwargs)) 

def environment_resolver(env, *wrappers):
    """ Resolve a gym environment.
    Usage in hydra config files:
    ```
        ${environment:'MyEnvironment-v0(<ARGS>,<KWARGS>)'}
    ```
    or
    ```
        ${environment:'MyEnvironment-v0'}
    ```
    Note: quotes 'name-version(...)' are required.

    Will convert to hydra object yaml: 
    ```
        _target_ : gymu.data.config._config.environment
        _args_: ['MyEnvironment-v0', 10]
        test: 10
    ```
    or  
    ```
        _target_ : gymu.data.config._config.environment
        _args_: ['MyEnvironment-v0']
    ```

    This will be later resolved by hydra to an environment factory for the specified environment.
    ```
        env_factory = hydra.utils.call(cfg.environment)     # MyEnvironment-v0 factory
        env = env_factory()                                 # new instance of MyEnvironment-v0
    ```

    Args:
        env (str): environment string to resolve
        wrappers (List[Any]): a list of wrappers that will be applied to the environment.
    Returns:
        DictConfig: hydra object yaml for the environment factory.
    """
    if len(wrappers) > 0:
        if OmegaConf.is_list(wrappers[0]):
            assert len(wrappers) == 1 # hmm..
            wrappers = wrappers[0]
    
    mod = ast.parse(env.strip())
    # TODO error message if doesnt contain Expr/BinOp
    binop = mod.body[0].value
    #print(ast.dump(binop))
    if isinstance(binop.left, ast.BinOp):
        namespace = binop.left.left.id + "/"
        name = binop.left.right.id
    else:
        namespace = ""
        name = binop.left.id
    if isinstance(binop.right, ast.Name):
        version = binop.right.id
        args = [f"{namespace}{name}-{version}"]
        kwargs = dict()
    else: # function call
        version = binop.right.func.id
        args = [f"{namespace}{name}-{version}"] + [n.value for n in binop.right.args]
        kwargs = {n.arg:n.value.value for n in binop.right.keywords}
    result = DictConfig(dict(_target_ = _fun_qual_name(environment), _args_=args, _wrappers_ = wrappers, **kwargs))
    return result

def spec(environment_factory):
    factory = hydra.utils.instantiate(environment_factory)
    get = _EnvironmentConfigGetter(factory)
    with get:
        return get()

class _EnvironmentConfigGetter:

    def __init__(self, env_factory):
        self.env_factory = env_factory
        self.env = None

    def __call__(self):
        spec = {k:v for k,v in self.env.spec.__dict__.items()}
        meta = self.get_environment_meta(self.env)
        config = dict(
            action_space=self.env.action_space, 
            observation_space=self.env.observation_space,
            spec=spec,
            meta=meta)
        # serialize action_space/observation_space etc.
        yaml_data = yaml.dump(config)
        yaml_data = _strip_yaml_tags(yaml_data)
        return OmegaConf.create(yaml_data)

    def __enter__(self):
        self.env = self.env_factory()
        self.env.reset() # some environments require this to prevent hanging...
 
    def __exit__(self, *args):
        if hasattr(self.env, "close"):
            self.env.close() # some environments should be closed... 
        self.env = None

    def get_environment_meta(self, env): # some additional environment stuff...
        meta = dict()
        if hasattr(env, "get_action_meanings"):
            meta['action_meanings'] = env.get_action_meanings()
        return meta


OmegaConf.register_new_resolver("get_class", hydra.utils.get_class)
OmegaConf.register_new_resolver("get_method", hydra.utils.get_method)
OmegaConf.register_new_resolver("environment", environment_resolver)
OmegaConf.register_new_resolver("wrapper", wrapper_resolver)
OmegaConf.register_new_resolver("transform", transform_resolver)
OmegaConf.register_new_resolver("spec", spec, use_cache=True)

class bind(partial):
    """ 
        An improved version of functools 'partial' which accepts Ellipsis (...) as a placeholder. 
        The 'dataset' argument in composition will not be filled in. 
    """
    def __call__(self, *args, **keywords):
        keywords = {**self.keywords, **keywords}
        iargs = iter(args)
        args = (next(iargs) if arg is ... else arg for arg in self.args)
        return self.func(*args, *iargs, **keywords)

def transform(fun, *args, **kwargs):
    return bind(fun, ..., *args, **kwargs)

def environment(env_id : str, *args : List[Any], _wrappers_ : List[Callable] = [], **kwargs : Dict[str, Any]): # refer to this in your configuration file
    """ Environment factory for use in hydra configs.
        Usage: 
        ```
            _target_ : gymu.data.config._config.environment
            _args_: ['MyEnvironment-v0', 10]
            _wrappers_ : []
            test: 10
        ```
        will create a factory for the given environment.
        ```
            env_factory = hydra.utils.call(cfg.environment)     # MyEnvironment-v0 factory
            env = env_factory()                                 # new instance of MyEnvironment-v0
        ```
    Args:
        env_id (str): environment id that follows the gym standard
        wrappers (List[Callable], optional): _description_. Defaults to [].
    """
    def make():
        env = gym.make(env_id, *args, **kwargs)
        for wrapper in _wrappers_: 
            env = wrapper(env)
        return env
    return make

# utilities

def _fun_qual_name(x):
    return f"{x.__module__}.{x.__qualname__}"

def _ast_expr_resolver(expr):
    if isinstance(expr.value, ast.Attribute): # its a fully qualified class/method name
        name = []
        for node in ast.walk(expr.value):
            if isinstance(node, ast.Name):
                name.append(node.id)
            elif isinstance(node, ast.Attribute):
                name.append(node.attr)
        name = ".".join(reversed(name))
        return SimpleNamespace(name = name, args=[], kwargs={})
    elif isinstance(expr.value, ast.Name):
        return SimpleNamespace(name = expr.value.id, args=[], kwargs={})

def _strip_yaml_tags(yaml_data): 
    result = []
    tab = "  " # check the file and update this to be consistent?
    for line in yaml_data.splitlines():
        idx = line.find("!")
        if idx > -1: # hydrafy
            result.append(line[:idx])
            result.append(tab +  f"_target_ : {line[idx:].replace('!', '')}")
        else:
            result.append(line)
    return '\n'.join(result)




