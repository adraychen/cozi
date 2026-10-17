"""Reproduce unrelated full-suite failures using HEAD factorial implementations.

This substitutes source in memory only; the working tree is never reset.
"""
import importlib
from pathlib import Path
import subprocess
import sys
import unittest

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
sys.path.insert(1, str(root / 'tests'))
for name in ('utils.factorial_anova_ui', 'utils.factorial_random_ui'):
    module = importlib.import_module(name)
    path = name.replace('.', '/') + '.py'
    source = subprocess.check_output(['git', 'show', 'HEAD:' + path], cwd=root, text=True, encoding='utf-8')
    exec(compile(source, str(root / path), 'exec'), module.__dict__)
suite = unittest.defaultTestLoader.loadTestsFromNames([
    'tests.test_fill_missing.FillMissingUITests.test_group_navigation_preview_apply_and_create',
    'tests.test_temporal_target_guidance.GuidanceUITests.test_workspace_preserves_controls_without_persisting_action',
])
result = unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(not result.wasSuccessful())
