from tensorflow.python.client import device_lib


def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU' or x.device_type == 'CPU']


if __name__ == '__main__':
    ds = get_available_devices()
    for d in ds:
        print(d)
