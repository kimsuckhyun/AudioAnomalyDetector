# AudioAnomalyDetector/src/run_all.py
import os
from src.main import main_enhanced_approach

def run_all_main_ids(
    data_root: str,
    save_dir: str = './best_models'
):
    os.makedirs(save_dir, exist_ok=True)
    for condition in sorted(os.listdir(data_root)):
        cond_path = os.path.join(data_root, condition)
        if not os.path.isdir(cond_path): continue
        for machine in sorted(os.listdir(cond_path)):
            mach_path = os.path.join(cond_path, machine)
            if not os.path.isdir(mach_path): continue
            for mid in sorted(os.listdir(mach_path)):
                mid_path = os.path.join(mach_path, mid)
                if not os.path.isdir(mid_path): continue
                print(f"Processing {condition}/{machine}/{mid}")
                try:
                    out = os.path.join(save_dir, f"best_{condition}_{machine}_{mid}.pth")
                    main_enhanced_approach(mach_path, mid, device='cuda')
                    os.rename('final_audio_anomaly_model.pth', out)
                except Exception as e:
                    print(f"Error at {condition}/{machine}/{mid}: {e}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--save', type=str, default='./best_models')
    args = parser.parse_args()
    run_all_main_ids(args.root, args.save)