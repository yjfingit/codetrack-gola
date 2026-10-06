# A very simple query language for json / yaml
#
#  following Pytorch nn.Module naming convention
#  JsonPath or Json Query is too complex for our use cases.
#
# examples:
#   config = {'a': {'b': {'c': 1}}}
#   json_query_get(config, 'a.b.c')
#   >>> 1
#   json_query_replace(config, 'a.b.c', 2)
#   config
#   >>> {'a': {'b': {'c': 2}}}
#   config = {'a': [{'b': {'c': 1}}, {'b': {'c': 2}}]}
#   json_query_get(config, 'a.0.b.c')
#   >>> 1
#   json_query_replace(config, 'a.0.b.c', 3)
#   config
#   >>> {'a': [{'b': {'c': 3}}, {'b': {'c': 2}}]}


def json_query_get(json_object, path: str):
    paths = path.split('.')
    for sub_path in paths:
        json_object = json_object[sub_path] if not sub_path.isdigit() else json_object[int(sub_path)]
    return json_object


def json_query_replace(json_object, path: str, value):
    # Zekai Shao: Skip the value if it is not exist
    paths = path.split('.')
    for sub_path in paths[:-1]:
        if sub_path.isdigit():
            index = int(sub_path)
            if not isinstance(json_object, list) or not 0 <= index < len(json_object):
                return
            json_object = json_object[index]
        else:
            if not isinstance(json_object, dict) or sub_path not in json_object:
                return
            json_object = json_object[sub_path]
    last = paths[-1]
    if last.isdigit():
        index = int(last)
        if isinstance(json_object, list) and 0 <= index < len(json_object):
            json_object[index] = value
    elif isinstance(json_object, dict) and last in json_object:
        json_object[last] = value


def json_query_retain(json_object, path: str, value):
    target_json_object = json_query_get(json_object, path)
    assert isinstance(target_json_object, dict)
    for key in list(target_json_object.keys()):
        if isinstance(value, (list, tuple)):
            if key not in value:
                del target_json_object[key]
        else:
            if key != value:
                del target_json_object[key]


def json_query_remove(json_object: dict, path: str, value):
    target_json_object = json_query_get(json_object, path)
    assert isinstance(target_json_object, (list, dict))

    if isinstance(target_json_object, dict):
        for key in list(target_json_object.keys()):
            if isinstance(value, (list, tuple)):
                if key in value:
                    del target_json_object[key]
            else:
                if key == value:
                    del target_json_object[key]
    elif isinstance(target_json_object, list):
        if isinstance(value, list):
            for v in value:
                target_json_object.remove(v)
        else:
            target_json_object.remove(value)


def json_query_merge(json_object, path: str, value):
    target_json_object = json_query_get(json_object, path)
    if isinstance(target_json_object, list):
        target_json_object.extend(value)
    elif isinstance(target_json_object, dict):
        target_json_object.update(value)
    else:
        raise RuntimeError(f'invalid target_json_object {target_json_object}')


def json_query_append(json_object, path: str, value):
    target_json_object = json_query_get(json_object, path)
    if isinstance(target_json_object, list):
        target_json_object.append(value)
    elif isinstance(target_json_object, str):
        target_json_object += value
    else:
        raise RuntimeError(f'invalid target_json_object {target_json_object}')
