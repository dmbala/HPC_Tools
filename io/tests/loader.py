"""Load the extensionless io tools as modules."""
import importlib.machinery
import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def load_tool(name):
    path = os.path.abspath(os.path.join(HERE, "..", name))
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod
