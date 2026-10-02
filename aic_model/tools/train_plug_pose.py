#!/usr/bin/env python3
"""Train and evaluate the plug-in-gripper keypoint model (aic_model/plug_pose.py).

Labels come from privileged training captures: `plug_tip.<camera>` holds the
plug-tip frame in each camera at the exposure (ground truth, collection only).
Each camera's crop is centred on the nominal grasp exactly as at runtime; a
wider margin is cached so training can shift the crop.

  cache:    captures -> <out>/cache_<type>.npz
  train:    cache -> checkpoint (one per plug type)
  evaluate: checkpoint + cache -> grasp error of triangulated, fitted poses
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aic_model import plug_pose as pp  # noqa: E402

MARGIN = 32
CACHE_PX = pp.CROP_PX+2*MARGIN


def nominal_grasps():
    from aic_model.policy_config import PolicyConfig
    return {t: pp.grasp_transform(h) for t, h in PolicyConfig._PLUG_OFFSETS.items()}


def validation_episode(episode_id, fraction=.2):
    return int(hashlib.sha256(episode_id.encode()).hexdigest()[:8], 16) < fraction*16**8


def select_captures(capture_dirs, plug_types, per_trial=None):
    """Labelled capture metadata of the given plug types; per_trial evenly spaced per episode.

    The arm is still during a collection trial, so its captures are near-duplicates;
    subsampling bounds the cache, which is held in memory."""
    episodes = {}
    for directory in capture_dirs:
        for meta_path in sorted(Path(directory).glob('**/captures/*.json')):
            d = json.loads(meta_path.read_text())
            labels = d.get('plug_tip', {})
            if d['task']['plug_type'] in plug_types and all('quat_camera' in labels.get(c, {}) for c in pp.CAMERAS):
                episodes.setdefault(d['episode_id'], []).append((meta_path, d))
    for items in episodes.values():
        if per_trial and len(items) > per_trial:
            items = [items[int(i)] for i in np.linspace(0, len(items)-1, per_trial).round()]
        yield from items


def build_cache(capture_dirs, out, plug_types=tuple(pp.PLUG_KEYPOINTS), per_trial=None):
    nominal = nominal_grasps()
    rows = {t: [] for t in plug_types}
    for meta_path, d in select_captures(capture_dirs, plug_types, per_trial):
        ptype = d['task']['plug_type']
        labels = d['plug_tip']
        images = np.load(meta_path.with_suffix('.npz'))
        tp = d['controller']['tcp_pose']
        T_base_tcp = pp.transform(pp.quat_matrix([tp['orientation'][k] for k in 'xyzw']),
                                  [tp['position'][k] for k in 'xyz'])
        crops, uvs, Ks, Ts, grasps, origins = [], [], [], [], [], []
        for cam in pp.CAMERAS:
            g, lab = d['camera_geometry'][cam], labels[cam]
            K = np.array(g['K'])
            T_tcp_cam = np.linalg.inv(T_base_tcp)@pp.transform(np.array(g['R_base_from_camera']),
                                                               g['t_base_from_camera'])
            T_cam_plug = pp.transform(pp.quat_matrix(lab['quat_camera']), lab['xyz_camera'])
            image = images[cam+'_image']
            height, width = image.shape[:2]
            x0, y0 = pp.crop_origin(K, T_tcp_cam, nominal[ptype], ptype, (width, height))
            padded = np.pad(image, ((MARGIN, MARGIN), (MARGIN, MARGIN), (0, 0)), mode='edge')
            crops.append(padded[y0:y0+CACHE_PX, x0:x0+CACHE_PX])
            origins.append((x0, y0))
            uvs.append(pp.project(K, T_cam_plug, pp.PLUG_KEYPOINTS[ptype])-[x0, y0])
            Ks.append(K)
            Ts.append(T_tcp_cam)
            grasps.append(T_tcp_cam@T_cam_plug)
        # The three cameras agree on the grasp to TF precision; keep the centre one.
        rows[ptype].append(dict(crops=np.stack(crops), uv=np.stack(uvs), K=np.stack(Ks), T_tcp_cam=np.stack(Ts),
                                grasp=grasps[1], origin=np.array(origins),
                                episode=d['episode_id'], val=validation_episode(d['episode_id'])))
    out.mkdir(parents=True, exist_ok=True)
    for ptype, items in rows.items():
        if not items:
            continue
        np.savez(out/f'cache_{ptype}.npz', **{k: np.stack([r[k] for r in items]) for k in items[0]})
        print(ptype, len(items), 'captures,', sum(r['val'] for r in items), 'validation,',
              len({r['episode'] for r in items}), 'episodes')


def heatmaps(uv, size, sigma):
    """Targets at stride CROP_PX/size; uv in crop pixels."""
    scale = pp.CROP_PX/size
    hm = (uv+.5)/scale-.5
    ys, xs = np.mgrid[0:size, 0:size]
    out = np.exp(-((xs[None]-hm[:, 0, None, None])**2+(ys[None]-hm[:, 1, None, None])**2)/(2*sigma**2))
    out[~np.isfinite(hm).all(axis=1)] = 0.
    return out.astype(np.float32)


def train(args):
    import torch
    from aic_model.landmark_network import LandmarkHeatmapNet
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    data = np.load(args.cache)
    ptype = args.cache.stem.split('_', 1)[1]
    crops, uv, val = data['crops'], data['uv'], data['val']
    train_idx = [(i, c) for i in np.flatnonzero(~val) for c in range(3)]
    val_idx = [(i, c) for i in np.flatnonzero(val) for c in range(3)]
    n_kp = len(pp.PLUG_KEYPOINTS[ptype])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LandmarkHeatmapNet(num_landmarks=n_kp).to(device)
    size = pp.CROP_PX//4
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    def batch(indices, augment):
        images, targets = [], []
        for i, c in indices:
            dx, dy = rng.integers(0, 2*MARGIN+1, 2) if augment else (MARGIN, MARGIN)
            image = crops[i, c, dy:dy+pp.CROP_PX, dx:dx+pp.CROP_PX].astype(np.float32)/255.
            if augment:
                image = np.clip(image*rng.uniform(.75, 1.25)+rng.uniform(-.08, .08)
                                + rng.normal(0, .01, image.shape), 0., 1.)
            images.append(image.astype(np.float32))
            targets.append(heatmaps(uv[i, c]-[dx-MARGIN, dy-MARGIN], size, args.sigma))
        x = pp.normalize(torch.as_tensor(np.stack(images), device=device).permute(0, 3, 1, 2))
        return x, torch.as_tensor(np.stack(targets), device=device)

    def loss_fn(pred, target):
        err = (torch.sigmoid(pred)-target)**2*(1.+20.*target)
        return err.mean()

    best = (np.inf, None)
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(len(train_idx))
        total = 0.
        for start in range(0, len(order), args.batch):
            x, y = batch([train_idx[j] for j in order[start:start+args.batch]], True)
            pred, _ = model(x)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss)*len(x)
        scheduler.step()
        model.eval()
        errors = []
        with torch.no_grad():
            from aic_model.landmark_network import heatmap_argmax
            for start in range(0, len(val_idx), args.batch):
                chunk = val_idx[start:start+args.batch]
                x, _ = batch(chunk, False)
                peaks = heatmap_argmax(torch.sigmoid(model(x)[0])).cpu().numpy()
                for (i, c), p in zip(chunk, peaks):
                    errors.append(np.linalg.norm((p+.5)*4-.5-uv[i, c], axis=1))
        errors = np.concatenate(errors) if errors else np.array([np.inf])
        median = float(np.median(errors))
        print(f'epoch {epoch+1} train={total/len(train_idx):.5f} val_px median={median:.2f} '
              f'p90={np.percentile(errors, 90):.2f}', flush=True)
        if median < best[0]:
            best = (median, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    torch.save({'plug_type': ptype, 'crop_px': pp.CROP_PX, 'keypoints': pp.PLUG_KEYPOINTS[ptype].tolist(),
                'min_confidence': args.min_confidence, 'state_dict': best[1],
                'val_median_px': float(best[0]), 'cache': str(args.cache), 'epochs': args.epochs}, args.output)
    print('saved', args.output, 'val median px', round(best[0], 2))


def evaluate(args):
    from aic_model.policy_config import PolicyConfig as C
    # Each key access to an .npz re-reads the whole array; load every array once.
    data = dict(np.load(args.cache))
    ptype = args.cache.stem.split('_', 1)[1]
    runtime = pp.load_plug_pose(args.checkpoint)
    nominal = nominal_grasps()[ptype]
    rows, episodes = [], {}
    for i in np.flatnonzero(data['val'] if not args.all else np.ones(len(data['val']), bool)):
        crops = [data['crops'][i, c, MARGIN:MARGIN+pp.CROP_PX, MARGIN:MARGIN+pp.CROP_PX] for c in range(3)]
        per_camera = [(uv+data['origin'][i, c], data['K'][i, c], data['T_tcp_cam'][i, c])
                      for c, (uv, _) in enumerate(runtime.keypoints(crops))]
        fit = pp.estimate_from_keypoints(ptype, per_camera, max_rms=C.GRASP_FIT_RMS_M,
                                         max_dropped=args.max_dropped)
        truth = data['grasp'][i]
        shift = pp.pose_difference(nominal, truth)
        shift = (shift[0]*1e3, np.degrees(shift[1]))
        # Every validation episode counts in the decision replay, fitted or not.
        episodes.setdefault(data['episode'][i], (truth, []))
        if fit is None:
            rows.append((data['episode'][i], np.nan, np.nan, np.nan, *shift, 0.))
            continue
        T, rms, points = fit
        episodes.setdefault(data['episode'][i], (truth, []))[1].append((T, rms))
        dt, dr = pp.pose_difference(T, truth)
        rows.append((data['episode'][i], dt*1e3, np.degrees(dr), rms*1e3, *shift, float(np.isnan(points).any())))
    t = np.array([r[1:] for r in rows], dtype=float)
    ok = np.isfinite(t[:, 0])
    print(f'{ptype}: {len(rows)} captures, fitted {ok.sum()}')
    if not ok.any():
        return
    for name, col in (('tip error mm', 0), ('rotation error deg', 1), ('fit rms mm', 2)):
        v = t[ok, col]
        print(f'  {name}: median {np.median(v):.2f} p90 {np.percentile(v, 90):.2f} max {v.max():.2f}')
    big = t[:, 4] > 3.
    fallback = t[:, 5] > 0
    for label, mask in (('nominal-like (<3 deg from _PLUG_OFFSETS)', ~big), ('shifted (>3 deg)', big),
                        ('fits with a dropped keypoint', fallback)):
        v = t[mask & ok]
        if len(v):
            print(f'  {label}: n={mask.sum()} fitted={len(v)} tip median {np.median(v[:, 0]):.2f} '
                  f'p90 {np.percentile(v[:, 0], 90):.2f} rot median {np.median(v[:, 1]):.2f} '
                  f'p90 {np.percentile(v[:, 1], 90):.2f}')
    decide(episodes, nominal)
    if args.report:
        Path(args.report).write_text(json.dumps([dict(zip(
            ('episode', 'tip_mm', 'rot_deg', 'rms_mm', 'shift_mm', 'shift_deg', 'fallback'), map(
                lambda x: x if isinstance(x, str) else float(x), r))) for r in rows], indent=1))


def decide(episodes, nominal):
    """Replay Policy._measure_grasp per validation episode: first GRASP_FRAMES fitted frames."""
    from aic_model.policy_config import PolicyConfig as C
    rows = []
    for truth, fits in episodes.values():
        frames = [T for T, rms in fits if rms <= C.GRASP_FIT_RMS_M][:C.GRASP_FRAMES]
        static = pp.pose_difference(nominal, truth)
        deep = nominal[2, 3]-truth[2, 3] > .006
        if len(frames) < C.GRASP_FRAMES:
            rows.append(('too few frames', static, static, deep))
            continue
        centre = np.median([T[:3, 3] for T in frames], axis=0)
        grasp = min(frames, key=lambda T: float(np.linalg.norm(T[:3, 3]-centre)))
        spread = [pp.pose_difference(grasp, T) for T in frames]
        agree = (max(d for d, _ in spread) <= C.GRASP_FRAME_SPREAD_M
                 and np.degrees(max(r for _, r in spread)) <= C.GRASP_FRAME_SPREAD_DEG)
        shift = pp.pose_difference(nominal, grasp)
        used = agree and (shift[0] > C.GRASP_APPLY_SHIFT_M or np.degrees(shift[1]) > C.GRASP_APPLY_SHIFT_DEG)
        result = pp.pose_difference(grasp, truth) if used else static
        rows.append(('used' if used else 'kept' if agree else 'disagree', static, result, deep))
    truly = lambda s: s[0] > C.GRASP_APPLY_SHIFT_M or np.degrees(s[1]) > C.GRASP_APPLY_SHIFT_DEG
    for label, select in (('shifted episodes', lambda r: truly(r[1])), ('normal episodes', lambda r: not truly(r[1])),
                          ('plug >6 mm deeper along TCP z (race-like)', lambda r: r[3])):
        sel = [r for r in rows if select(r)]
        if not sel:
            continue
        counts = {k: sum(r[0] == k for r in sel) for k in ('used', 'kept', 'disagree', 'too few frames')}
        before = np.array([(s[0]*1e3, np.degrees(s[1])) for _, s, _, _ in sel])
        after = np.array([(s[0]*1e3, np.degrees(s[1])) for _, _, s, _ in sel])
        print(f'  {label} (n={len(sel)}): {counts}')
        print(f'    grasp error with the fixed offsets: tip median {np.median(before[:, 0]):.2f} max '
              f'{before[:, 0].max():.2f} mm, rot median {np.median(before[:, 1]):.2f} max {before[:, 1].max():.2f} deg')
        print(f'    grasp error after the decision:    tip median {np.median(after[:, 0]):.2f} max '
              f'{after[:, 0].max():.2f} mm, rot median {np.median(after[:, 1]):.2f} max {after[:, 1].max():.2f} deg')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('cache')
    p.add_argument('captures', nargs='+', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--types', nargs='+', choices=tuple(pp.PLUG_KEYPOINTS), default=tuple(pp.PLUG_KEYPOINTS),
                   help='cache one plug type per call to bound memory')
    p.add_argument('--per-trial', type=int, help='evenly spaced captures kept per episode')
    p = sub.add_parser('train')
    p.add_argument('cache', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--sigma', type=float, default=1.5)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--min-confidence', type=float, default=.3)
    p = sub.add_parser('evaluate')
    p.add_argument('cache', type=Path)
    p.add_argument('checkpoint', type=Path)
    p.add_argument('--all', action='store_true')
    p.add_argument('--report')
    p.add_argument('--max-dropped', type=int, default=pp.MAX_DROPPED_KEYPOINTS, help='keypoints the fit may leave out')
    args = parser.parse_args()
    {'cache': lambda: build_cache(args.captures, args.out, args.types, args.per_trial), 'train': lambda: train(args),
     'evaluate': lambda: evaluate(args)}[args.action]()


if __name__ == '__main__':
    main()
