"""
Fix script for: ModuleNotFoundError: No module named 'data_pipeline'

WHAT IS HAPPENING
-----------------
Your project has this layout:

    forest-fire-prediction/
        data_pipeline.py
        model_explanations.py
        models/
            xgboost_model.py     <- imports `from data_pipeline import ...`
            random_forest.py
            neural_network.py
        spread_model/
        evaluation/
        ...

When you run `python3 models/xgboost_model.py`, Python sets the import root
to `models/`. From there, `data_pipeline` is one directory up - Python does
not look up the directory tree. So it cannot find data_pipeline.py and you
get the ModuleNotFoundError.

This is not a bug in the code. It is how Python's import system works. The
fix is one of three options.


OPTION 1: Run from the project root with -m (RECOMMENDED, no code change)
-------------------------------------------------------------------------
Always run scripts from the project root, using the -m module flag:

    cd forest-fire-prediction
    python3 -m models.xgboost_model

The -m flag tells Python to treat models.xgboost_model as a module,
which makes the project root the import root, which means the import
of data_pipeline succeeds.

Tell PyCharm or VS Code to do this too:
- PyCharm: Run > Edit Configurations > set Working Directory to project root
  and use "module name" instead of "script path"
- VS Code: in launch.json, set "module": "models.xgboost_model" and
  "cwd": "${workspaceFolder}"


OPTION 2: Add __init__.py files (small code change, makes imports proper)
--------------------------------------------------------------------------
Add empty __init__.py files to each subdirectory that contains Python files
you want to import:

    touch forest-fire-prediction/__init__.py
    touch forest-fire-prediction/models/__init__.py
    touch forest-fire-prediction/spread_model/__init__.py
    touch forest-fire-prediction/evaluation/__init__.py

This makes them proper Python packages. With this in place, you can still
run from the project root with `python3 -m models.xgboost_model`. The
__init__.py files are also required if you ever package the project for
distribution.


OPTION 3: sys.path hack inside each model file (LAST RESORT)
------------------------------------------------------------
If you cannot change how you run the script, add this to the top of
each models/*.py file BEFORE any project imports:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from data_pipeline import FireDataPipeline   # now works

This forces Python to add the project root to its import path at runtime.
It works but it is fragile (every script needs the same boilerplate) and
it is not what you want long-term.


WHAT THIS SCRIPT DOES
---------------------
This script applies OPTION 2: it creates the __init__.py files so the
project becomes a proper Python package layout. Run it from the project
root:

    python3 fix_imports.py

After running it once, you can run any model file with:

    python3 -m models.xgboost_model

without further changes.
"""
from pathlib import Path
import sys


# Folders that should be Python packages. Add to this list if you have
# more subdirectories with code you import.
PACKAGE_DIRS = [
    'models',
    'spread_model',
    'evaluation',
    'utils',
]


def main():
    project_root = Path.cwd()

    # Sanity check: are we actually in the project root?
    sentinels = ['data_pipeline.py', 'models']
    missing = [s for s in sentinels if not (project_root / s).exists()]
    if missing:
        print(f'ERROR: This does not look like the project root.')
        print(f'  Working directory: {project_root}')
        print(f'  Expected to find: {sentinels}')
        print(f'  Missing:          {missing}')
        print()
        print('Run this script from the forest-fire-prediction directory.')
        return 1

    created = []
    skipped = []

    for sub in PACKAGE_DIRS:
        sub_path = project_root / sub
        if not sub_path.is_dir():
            continue
        init_path = sub_path / '__init__.py'
        if init_path.exists():
            skipped.append(str(init_path.relative_to(project_root)))
        else:
            init_path.write_text(
                f'"""Package marker for {sub}/. Created by fix_imports.py."""\n'
            )
            created.append(str(init_path.relative_to(project_root)))

    # Also a top-level __init__.py is optional but harmless
    top_init = project_root / '__init__.py'
    if not top_init.exists():
        top_init.write_text('"""Forest fire prediction project."""\n')
        created.append('__init__.py')

    print('Done.')
    if created:
        print(f'\nCreated {len(created)} __init__.py file(s):')
        for p in created:
            print(f'  + {p}')
    if skipped:
        print(f'\nSkipped {len(skipped)} (already existed):')
        for p in skipped:
            print(f'  = {p}')
    print()
    print('Now run model files from the project root with the -m flag:')
    print('  python3 -m models.xgboost_model')
    print('  python3 -m models.random_forest')
    print('  python3 -m models.neural_network')
    return 0


if __name__ == '__main__':
    sys.exit(main())
