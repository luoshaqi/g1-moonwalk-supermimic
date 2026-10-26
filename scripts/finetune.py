"""Transfer E009 network weights; intentionally reset optimizers, replay and sampler.

Uses MIT-licensed SuperMimic_Dyn training helpers; see THIRD_PARTY_NOTICES.md.
"""
from pathlib import Path
import argparse
import json
import time
from common import ROOT, configure

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--motion-dir', type=Path, default=ROOT/'data')
    parser.add_argument('--log-root', type=Path, default=ROOT/'outputs/logs')
    parser.add_argument('--run-name', default='moonwalk')
    parser.add_argument('--iterations', type=int, default=3000)
    parser.add_argument('--num-envs', type=int, default=1024)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    configure()
    import torch
    from run_single_gpu import build_parser, build_config, ensure_run_dir_available, parameter_snapshot, parameter_change
    from supermimic.training.loop import Trainer
    if args.iterations < 70:
        parser.error('At least 70 iterations are required to pass replay warmup and verify updates.')
    cfg_args = build_parser().parse_args([
        '--mode','train','--npz-dir',str(args.motion_dir.resolve()),
        '--robot-variant','g1_no_plastic_hands','--anchor-mode','full',
        '--run-name',args.run_name,'--log-root',str(args.log_root.resolve()),
        '--num-envs',str(args.num_envs),'--batch-size',str(args.batch_size),
        '--no-compile','--iterations',str(args.iterations),'--seed',str(args.seed)])
    cfg = build_config(cfg_args)
    cfg.schedule.max_hours = None
    cfg.schedule.checkpoint_interval = 500
    cfg.wm_trainer.lr = cfg.policy_trainer.lr = cfg.policy_trainer.critic_lr = 5e-5
    ensure_run_dir_available(cfg)
    torch.set_num_threads(4)
    t0 = time.monotonic()
    trainer = Trainer(cfg)
    summary = {'source_checkpoint': str(args.checkpoint), 'motion': 'CMU 90_32 G1 retarget v2',
               'transfer': 'all three networks; fresh optimizers/replay/sampler',
               'device': torch.cuda.get_device_name(0)}
    try:
        # Only load checkpoints obtained from trusted sources (PyTorch pickle format).
        ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        summary['source_iteration'] = ck.get('iteration', 40000)
        modules = {'world_model': trainer.wm, 'actor': trainer.policy, 'critic': trainer.critic}
        for name, key in [('world_model','wm'),('actor','policy'),('critic','critic')]:
            modules[name].load_state_dict(ck[key], strict=True)
        del ck
        initial = {k: parameter_snapshot(m) for k,m in modules.items()}
        print('TRANSFER_LOADED: strict E009 network load; fresh optimizer/replay/sampler', flush=True)
        trainer.save('ckpt_initial.pt')
        torch.cuda.reset_peak_memory_stats()
        trainer.run()
        trainer.save(f'ckpt_{trainer.iteration:07d}.pt')
        changes = {k: parameter_change(m, initial[k]) for k,m in modules.items()}
        if trainer.iteration != args.iterations or trainer.num_updates <= 0:
            raise RuntimeError('Training did not finish with optimizer updates')
        if not all(v['finite'] and v['changed_tensors'] > 0 for v in changes.values()):
            raise RuntimeError('A trained network was unchanged or nonfinite')
        summary.update(execution='completed', iteration=trainer.iteration,
                       num_updates=trainer.num_updates, parameter_changes=changes)
    except BaseException as e:
        summary.update(execution='failed', error=repr(e))
        raise
    finally:
        summary.update(elapsed_seconds=time.monotonic()-t0,
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       peak_reserved_bytes=torch.cuda.max_memory_reserved())
        (trainer.log_dir/'finetune_summary.json').write_text(json.dumps(summary,indent=2))
        trainer.env.close()
        if trainer.writer: trainer.writer.close()
        print(json.dumps(summary,indent=2),flush=True)

if __name__ == '__main__': main()
