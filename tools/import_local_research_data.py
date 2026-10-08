"""Copy authorized local research inputs into ignored private_data; never upload."""
import argparse
import json
from pathlib import Path
import shutil

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',required=True,type=Path)
    args=parser.parse_args()
    src=args.source_root.resolve()
    target=Path(__file__).resolve().parents[1]/'private_data'
    inputs={
        'reference_lens.json':'results/overnight_rms_search/round_0_original/manufacturing_lens.json',
        'recovered_seed_lens.json':'results/mtf_yield_curriculum/seeds/1_stressed_search_start/manufacturing_lens.json',
        'main_seed_lens.json':'results/high_cost_curriculum/round_9/p_c9ea076c59/trial_5_s2_h_lak7a/nominal_recovery/manufacturing_lens.json',
        'relative_prices.csv':'configs/cdgm_relative_prices.csv',
    }
    for name in ['CDGM.AGF','SCHOTT.AGF','MISC.AGF','PLASTIC2022.AGF','materials_data.json']:
        inputs['material/'+name]='deeplens/optics/material/'+name
    missing=[rel for rel in inputs.values() if not (src/rel).is_file()]
    if missing: raise FileNotFoundError('Missing source inputs: '+', '.join(missing))
    for dest,rel in inputs.items():
        path=target/dest; path.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src/rel,path)
    print(f'Imported {len(inputs)} files into ignored private_data. Do not upload these inputs.')

if __name__=='__main__': main()
