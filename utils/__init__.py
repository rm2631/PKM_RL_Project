from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    ProgressBarCallback,
)


def print_section(text):
    print("" * 80)
    print("=" * 80)
    print(text)
    print("=" * 80)
    print("" * 80)


def handle_callbacks(is_test):
    if is_test:
        return []
    else:
        return [
            ProgressBarCallback(),
        ]
