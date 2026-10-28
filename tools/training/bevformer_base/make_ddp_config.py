"""Add a three-GPU mini preflight config without modifying single-GPU evidence."""

import argparse
from pathlib import Path


def write_config(task_root):
    task = Path(task_root).expanduser().resolve()
    directory = task / "configs"
    if not (directory / "base_smoke.py").is_file():
        raise FileNotFoundError("Generate base_smoke.py with make_configs.py first.")
    text = f'''# Three GPU preflight ONLY: four real targets repeated three times.
# Eight stored context/target frames and full Base architecture remain unchanged.
_base_ = ['./base_smoke.py']
custom_imports = dict(imports=['base_training_support', 'ddp_preflight'],
                      allow_failed_imports=False)
base_world_size = 3
data = dict(train=dict(type='RepeatedBaseSmokeDataset', repeats=3))
optimizer = dict(lr=7.5e-5)
work_dir = {str(task / 'work_dirs/ddp-smoke')!r}
custom_hooks = [
    dict(type='TransferWeight', priority='LOWEST'),
    dict(type='SyncHistoryBeforeRunHook', priority='LOWEST'),
    dict(type='BaseArchitectureCheckHook', expected_world_size=3,
         expected_version='v1.0-mini', expected_train_samples=12, priority='LOWEST'),
    dict(type='BaseTrainingHealthHook', interval=1, priority='LOWEST'),
    dict(type='BaseDDPPreflightHook', expected_steps=4, expected_world_size=3,
         priority='LOWEST')]
'''
    path = directory / "base_ddp_smoke.py"
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise FileExistsError("Refusing to overwrite a different configuration: " + str(path))
    path.write_text(text, encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", type=Path, required=True)
    args = parser.parse_args()
    print(write_config(args.task_root))


if __name__ == "__main__":
    main()
