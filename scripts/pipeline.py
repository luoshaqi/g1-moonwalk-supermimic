"""Fine-tune, evaluate three checkpoints, then render the best closed-loop rollout."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from common import ROOT, configure

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,default=ROOT/'checkpoints/ckpt_0040000.pt')
    p.add_argument('--motion',type=Path,default=ROOT/'data/cmu90_32_moonwalk.npz')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/moonwalk')
    p.add_argument('--stage',choices=['all','evaluate'],default='all',help='evaluate reuses checkpoints and never starts training')
    p.add_argument('--run-dir',type=Path,help='Existing run directory, required for --stage evaluate')
    p.add_argument('--eval-iterations',type=int,nargs='+',default=[1000,2000,3000])
    p.add_argument('--backend',choices=['egl','osmesa','glfw'],default='egl')
    args = p.parse_args()
    if args.stage == 'evaluate' and args.run_dir is None: p.error('--stage evaluate requires --run-dir')
    upstream = configure()
    art=args.output.resolve();motion=args.motion.resolve();checkpoint=args.checkpoint.resolve()
    if art.exists() and any(art.iterdir()): p.error('Use a new --output directory; existing results are preserved.')
    if motion.stem != 'cmu90_32_moonwalk': p.error('Motion file must be named cmu90_32_moonwalk.npz')
    if not motion.is_file() or not checkpoint.is_file(): p.error('Motion or pretrained checkpoint is missing')
    run_dir=args.run_dir.resolve() if args.run_dir else art/'logs/moonwalk'
    iterations=args.eval_iterations
    if args.stage=='all' and iterations != [1000,2000,3000]: p.error('Default training evaluates 1000/2000/3000; use evaluate for other runs')
    delivery=art/'delivery';delivery.mkdir(parents=True)
    env=os.environ.copy()
    env['PYTHONPATH']=os.pathsep.join(map(str,[upstream/'src',upstream/'vendor/mjlab/src',upstream/'scripts']))
    env['MUJOCO_GL']=args.backend
    state={'started_unix':time.time(),'motion_sha256':hashlib.sha256(motion.read_bytes()).hexdigest()}
    def status(name,**kwargs):
        state.update(status=name,updated_unix=time.time(),**kwargs)
        tmp=art/'status.tmp.json';tmp.write_text(json.dumps(state,indent=2));tmp.replace(art/'status.json')
        print('PIPELINE_STATUS',name,flush=True)
    def run(cmd): subprocess.run([sys.executable,*map(str,cmd)],cwd=upstream,env=env,check=True)
    def evaluate(ck,label):
        dest=art/label
        run([upstream/'scripts/evaluate_hard_suite.py','--ckpt',ck,'--manifest',art/'manifest.csv',
             '--device','cuda:0','--num-envs','1','--seed','0','--out',dest/'metrics.json',
             '--trajectory-dir',dest/'trajectories','--trajectory-clips','cmu90_32_moonwalk'])
        return json.loads((dest/'metrics.json').read_text())['clips'][0]
    def render(label,iteration,name):
        run([ROOT/'scripts/render_follow.py','--trajectory',art/label/'trajectories/cmu90_32_moonwalk.npz',
             '--out',delivery/name,'--robot-variant','g1_no_plastic_hands','--layout','overlay',
             '--iteration',iteration,'--reference-alpha','.2','--backend',args.backend,
             '--width','1280','--height','720','--fps','25','--max-seconds','10'])
    try:
        with (art/'manifest.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['name','npz','class','split','source','category_group'])
            w.writerow(['cmu90_32_moonwalk',motion,'moonwalk','train','CMU 90_32 G1 retarget v2','dance'])
        status('baseline_evaluation');baseline=evaluate(checkpoint,'baseline')
        if args.stage=='all':
            status('finetuning')
            run([ROOT/'scripts/finetune.py','--checkpoint',checkpoint,'--motion-dir',motion.parent,
                 '--log-root',run_dir.parent,'--run-name',run_dir.name])
        status('checkpoint_evaluation')
        results={str(n):evaluate(run_dir/f'ckpt_{n:07d}.pt',f'finetune_{n}') for n in iterations}
        def score(x):return (not x['failed'],x['completion_ratio'],x['full_clip_normalized_return'])
        best=max(results,key=lambda n:score(results[n]))
        status('rendering',selected_finetune_iteration=int(best))
        render('baseline',40000,'baseline.mp4');render(f'finetune_{best}',int(best),'finetuned.mp4')
        report={'baseline':baseline,'finetuned_checkpoints':results,'selected_finetune_iteration':int(best),
                'selection_rule':'Prefer nonfailed, then completion ratio, then normalized return; among fine-tuned checkpoints.',
                'finetuned_better_than_baseline_by_selection_rule':score(results[best])>score(baseline),
                'scope':'Training-motion fitting in simulation; not generalization or real-robot validation.',
                'motion_sha256':state['motion_sha256'],
                'finetune_summary':json.loads((run_dir/'finetune_summary.json').read_text())}
        (delivery/'comparison.json').write_text(json.dumps(report,indent=2))
        status('complete',finished_unix=time.time())
    except BaseException as e:
        status('failed',error=repr(e),traceback=traceback.format_exc());raise

if __name__=='__main__': main()
