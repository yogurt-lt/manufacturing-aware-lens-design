"""Portable entry point: validate inputs, then run search with progress logging."""
import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent

def check(config_path):
    import yaml
    settings = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    base = yaml.safe_load(Path(settings['base_config']).read_text(encoding='utf-8'))
    if settings.get('material_data_dir'):
        directory = Path(settings['material_data_dir']).resolve()
        if not directory.is_dir():
            raise FileNotFoundError('Research catalogs missing. See docs/DATA.md and import your authorized inputs.')
        os.environ['MANUFACTURING_MATERIAL_DIR'] = str(directory)
    files = [base['glass_catalog'], settings['reference_lens_json']]
    files += [item['path'] for item in settings['seed_lenses']]
    for filename in files:
        if not Path(filename).is_file():
            raise FileNotFoundError(f'Missing input: {filename}; see docs/DATA.md')
    import torch
    from deeplens.geolens import GeoLens
    from deeplens.optics.materials import Material
    expert = importlib.import_module('17_manufacturing_expert')
    prices = expert.prices_from_csv(base['glass_catalog'])
    for filename in files[1:]:
        lens = json.loads(Path(filename).read_text(encoding='utf-8'))
        for surface in lens['surfaces']:
            name = surface['mat2'].lower()
            Material(name)
            if name not in {'air','vacuum','occluder'} and name not in prices:
                raise ValueError(f'Missing cost entry for {name}')
    device = settings.get('device','auto')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; install compatible PyTorch or use device: auto.')
    proposals,_ = expert.propose_real_glasses(lens,prices,settings['proposal'])
    print(f'Checks passed | Python {sys.version.split()[0]} | torch {torch.__version__} | CUDA {torch.cuda.is_available()} | {len(proposals)} material proposals',flush=True)
    return settings

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/demo_search.yml')
    parser.add_argument('--check',action='store_true',help='Validate inputs and dependencies; do not optimize.')
    parser.add_argument('--no-monitor',action='store_true')
    parser.add_argument('--interval',type=float,default=15)
    parser.add_argument('--replot-only',action='store_true')
    args=parser.parse_args()
    os.chdir(ROOT)
    config_path=Path(args.config).resolve()
    settings=check(config_path)
    if args.check: return
    cmd=[sys.executable,'-u','20_hammer_beam_search.py','--config',str(config_path)]
    if args.replot_only: cmd.append('--replot-only')
    if args.no_monitor or args.replot_only:
        raise SystemExit(subprocess.call(cmd))
    folder=ROOT/'results/runtime_logs'; folder.mkdir(parents=True,exist_ok=True)
    stamp=time.strftime('%Y%m%d_%H%M%S')
    outpath=folder/f'{stamp}.out.log'; errpath=folder/f'{stamp}.err.log'
    print(f'Output: {settings["output_dir"]}\nLog: {outpath}',flush=True)
    with outpath.open('w',encoding='utf-8') as stdout, errpath.open('w',encoding='utf-8') as stderr:
        process=subprocess.Popen(cmd,stdout=stdout,stderr=stderr)
        try:
            while process.poll() is None:
                subprocess.run([sys.executable,'scripts/show_search_progress.py','--config',str(config_path)],check=False)
                for line in outpath.read_text(encoding='utf-8',errors='replace').splitlines()[-5:]: print(line)
                sys.stdout.flush()
                # Small polls let Ctrl+C respond promptly without orphaning training.
                deadline=time.monotonic()+max(2,args.interval)
                while process.poll() is None and time.monotonic()<deadline: time.sleep(0.25)
        except KeyboardInterrupt:
            process.terminate(); process.wait()
            print('Search stopped; completed stages remain available for resume.')
            raise SystemExit(130)
    code=process.returncode
    print(outpath.read_text(encoding='utf-8',errors='replace')[-3000:])
    if code:
        print(errpath.read_text(encoding='utf-8',errors='replace')[-5000:])
        raise SystemExit(code)
    print('Search completed successfully.',flush=True)

if __name__=='__main__': main()
