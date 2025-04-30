def get_boolean(config, key):
    if isinstance(config, dict) and key in config:
        return config[key] is True
    return False

def is_suppressed(weights):
    return get_boolean(weights, '_suppress')
