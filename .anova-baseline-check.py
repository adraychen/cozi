"""Verify two unrelated suite failures against original ANOVA renderers."""
from pathlib import Path
import shutil, subprocess, sys, tempfile
sys.stdout.reconfigure(encoding='utf-8')
root=Path.cwd()
with tempfile.TemporaryDirectory(prefix='anova-baseline-',dir=root) as tmp:
    dest=Path(tmp)
    for name in ['utils','tests','.streamlit','data']:
        shutil.copytree(root/name,dest/name,ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(root/'app.py',dest/'app.py')
    for name in ['utils/anova_ui.py','utils/factorial_anova_ui.py']:
        (dest/name).write_bytes(subprocess.check_output(['git','show',':'+name],cwd=root))
    for pattern in ['test_fill_missing.py','test_temporal_target_guidance.py']:
        result=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','tests','-p',pattern],cwd=dest,capture_output=True,text=True,encoding='utf-8',errors='replace')
        print(pattern, 'exit', result.returncode)
        print(result.stdout+result.stderr)
