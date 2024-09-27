import importlib


def import_pipeline(
        pipeline_name: str, module: str = 'src.pipelines') -> callable:
    """
    Given a pipeline module name, this function imports and returns the
    imported pipeline callable.

    Args:
        pipeline_name (str): pipeline module name to import.
        module (optional, str): base pipelines module. Default value is
            src.pipelines.

    Returns:
        (callable): pipeline module.
    """
    return importlib.import_module(f'{module}.{pipeline_name}')


def ensure_input_list(obj: object) -> list:
    """
    Ensure that input is a list; if not, return one.

    Args:
        obj: Any object.

    Returns:
        lst: Returned list.

    """
    if isinstance(obj, list):
        lst = obj
    elif not obj:
        lst = []
    else:
        lst = [obj]
    return lst


def perform_dict_union_recursively(
        dict_x: dict, dict_y: dict) -> dict:
    """
    Perform a union of dictionaries, which means to take keys and values from
    both of them, without doing any replacement.
    This assumes that there are no equal low-level keys between both objects.
    The way it works is pretty simple:
    - It creates a new dictionary with unique keys from both inputs
    - Recursively call the function for common keys (which can be considered
      new dictionaries with their own keys)

    Args:
        dict_x: One of the dictionaries to merge.
        dict_y: The other dictionary to merge.

    Returns:
        Pure union of input dictionaries.

    Raises:
        AttributeError in case inputs share a key which is not another dict.
    """
    output = {}
    x_keys, y_keys = set(dict_x.keys()), set(dict_y.keys())
    shared_keys = list(x_keys.intersection(y_keys))
    unique_x = list(x_keys.difference(set(shared_keys)))
    unique_y = list(y_keys.difference(set(shared_keys)))
    for key in unique_x:
        output[key] = dict_x[key]
    for key in unique_y:
        output[key] = dict_y[key]
    for key in shared_keys:
        if not isinstance(dict_x[key], dict) \
           or not isinstance(dict_y[key], dict):
            raise AttributeError(
                f'Pure union cannot be made due to key ["{key}"]. '
                'Please review both dictionaries.')
        else:
            output[key] = perform_dict_union_recursively(
                dict_x[key], dict_y[key])
    return output


def import_library(
        module: str, params: dict = None) -> object:
    """
    Imports the module specified and creates a new callable with specific
    parameters.

    Args:
        module: Module name.
        params: Parameters for the specific module initialization.

    Returns:
        Callable of imported module with parameters.
    """
    library = '.'.join(module.split('.')[:-1])
    name = module.split('.')[-1]
    imported_module = importlib.import_module(library)

    if params is None:
        params = {}

    inst = getattr(imported_module, name)(**params)
    return inst
