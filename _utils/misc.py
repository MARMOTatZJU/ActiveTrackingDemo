import importlib
import imp

def import_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module



# def import_module(name, path):
#     module = imp.load_source(name, path)

#     return module
