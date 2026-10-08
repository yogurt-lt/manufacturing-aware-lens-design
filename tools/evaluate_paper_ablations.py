"""Evaluate completed paper ablations with the authorized external catalog."""
import os
from pathlib import Path
import runpy
import sys

root=Path(__file__).resolve().parents[1]
os.chdir(root)
sys.path.insert(0,str(root))
catalog=root/'private_data/material'
if not catalog.is_dir(): raise FileNotFoundError('Import authorized research data first; see docs/DATA.md')
os.environ['MANUFACTURING_MATERIAL_DIR']=str(catalog)
runpy.run_path(str(root/'scripts/evaluate_formal_ablations.py'),run_name='__main__')
runpy.run_path(str(root/'scripts/bootstrap_formal_ablation.py'),run_name='__main__')
