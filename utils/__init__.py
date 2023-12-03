import os


def print_section(text):
    print("" * 80)
    print("=" * 80)
    print(text)
    print("=" * 80)
    print("" * 80)


def _get_path(prefix, reset_id=None):
    path = os.path.join("artifacts", os.environ["run_name"], prefix)
    if reset_id:
        path = os.path.join(path, reset_id)
    if not os.path.exists(path):
        os.makedirs(path)
    return path
