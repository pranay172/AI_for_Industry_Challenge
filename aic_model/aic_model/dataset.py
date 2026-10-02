"""Shared capture provenance and scene-disjoint dataset validation."""
from pathlib import Path
import hashlib
import random


def training_record(args, trainer):
    """Record the exact initialization and training implementation for a candidate."""
    import platform
    import torch
    source = Path(trainer).resolve()
    package = Path(__file__).resolve().parent
    files = [source, package/'dataset.py', package/'landmark_network.py', package/'vision_runtime.py', package/'rail_conditioning.py']
    return {
        'arguments': {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
        'source_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
        'initialize_sha256': hashlib.sha256(Path(args.initialize).read_bytes()).hexdigest()
                              if args.initialize else None,
        'python': platform.python_version(), 'torch': str(torch.__version__),
        'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def save_checkpoint(checkpoint, destination):
    """Publish a complete candidate atomically, preserving the previous best on failure."""
    import os
    import tempfile
    import torch
    destination = Path(destination)
    fd, temporary = tempfile.mkstemp(prefix=destination.name + '.', suffix='.tmp',
                                     dir=destination.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            torch.save(checkpoint, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def resolve_capture(raw_path, manifest):
    """Resolve portable paths; legacy relocation must be explicitly authored."""
    path = Path(raw_path).expanduser()
    if not raw_path:
        raise ValueError('Label row has no capture path')
    return str((path if path.is_absolute() else Path(manifest).resolve().parent / path).resolve())


def capture_group(row, manifest):
    # Scene ID wins: repeated episodes of the same scene stay together.
    explicit = row.get('scene_id') or row.get('episode_id')
    if explicit:
        return str(explicit)
    # Conservative legacy fallback: never split adjacent frames within a folder.
    return 'legacy-directory:' + str(Path(resolve_capture(row.get('npz_path', ''), manifest)).parent)


def grouped_split(samples, val_fraction, seed):
    if not 0 < val_fraction < 1:
        raise ValueError('val_fraction must be strictly between 0 and 1')
    groups = sorted({sample.group_id for sample in samples})
    if len(groups) < 2:
        raise ValueError('Need at least two independent scene/episode groups; frame splitting is prohibited')
    random.Random(seed).shuffle(groups)
    count = min(len(groups) - 1, max(1, round(len(groups) * val_fraction)))
    validation = set(groups[:count])
    return ([i for i, s in enumerate(samples) if s.group_id not in validation],
            [i for i, s in enumerate(samples) if s.group_id in validation])


def validate_captures(samples):
    import numpy as np
    by_path = {}
    for sample in samples:
        by_path.setdefault(sample.npz_path, set()).add(sample.image_key)
    missing = [path for path in by_path if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f'{len(missing)} capture files missing; first: {missing[0]}')
    expected_hashes = {}
    crop_boxes = {}
    for sample in samples:
        box = getattr(sample, 'crop_box', None)
        if box is not None:
            crop_boxes.setdefault((sample.npz_path,sample.image_key), []).append(box)
            expected_hashes.setdefault(sample.npz_path, set()).add(sample.crop_capture_sha256)
    for path, expected in expected_hashes.items():
        if len(expected) != 1 or hashlib.sha256(Path(path).read_bytes()).hexdigest() not in expected:
            raise ValueError(f'Capture changed since runtime crop preparation: {path}')
    for path, keys in by_path.items():
        with np.load(path, allow_pickle=False) as capture:
            for key in keys:
                if key not in capture:
                    raise ValueError(f'{path}: missing camera image {key}')
                image = capture[key]
                if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                    raise ValueError(f'{path}:{key}: expected HxWx3 uint8 RGB')
                for x0,y0,x1,y1 in crop_boxes.get((path,key), []):
                    if not (0 <= x0 < x1 <= image.shape[1] and 0 <= y0 < y1 <= image.shape[0]):
                        raise ValueError(f'Runtime crop exceeds image dimensions: {path}:{key}')


def split_manifest(samples, train_indices, val_indices, manifest, seed):
    def partition(indices):
        return {'groups': sorted({samples[i].group_id for i in indices}),
                'captures': sorted({samples[i].npz_path for i in indices}),
                'rows': len(indices)}
    train, val = partition(train_indices), partition(val_indices)
    if set(train['captures']) & set(val['captures']):
        raise ValueError('A capture is assigned to more than one group/partition')
    return {'schema_version': 1, 'seed': seed,
            'labels_sha256': hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
            'train': train, 'validation': val}


def assert_independent(reference, evaluation):
    if ({s.group_id for s in reference} & {s.group_id for s in evaluation}
            or {s.npz_path for s in reference} & {s.npz_path for s in evaluation}):
        raise ValueError('Evaluation overlaps training/validation groups or captures')


def explicit_validation_split(training, validation, training_manifest, validation_manifest, seed):
    """Use preassigned scene partitions without shuffling validation into training."""
    if not training or not validation:
        raise ValueError('Explicit training and validation partitions must both be nonempty')
    assert_independent(training, validation)
    samples = list(training) + list(validation)
    train_indices = list(range(len(training)))
    val_indices = list(range(len(training), len(samples)))
    record = split_manifest(samples, train_indices, val_indices, training_manifest, seed)
    record.update(partition_method='explicit_manifests',
                  validation_labels_sha256=hashlib.sha256(Path(validation_manifest).read_bytes()).hexdigest())
    return samples, train_indices, val_indices, record


def load_rgb(sample):
    """A missing image is an error, never a black image with positive labels."""
    import numpy as np
    from PIL import Image
    with np.load(sample.npz_path, allow_pickle=False) as capture:
        image = capture[sample.image_key]
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f'Invalid RGB capture: {sample.npz_path}:{sample.image_key}')
        return Image.fromarray(image).convert('RGB')


def target_crop(image, points, visible, augment=False):
    """Crop with target context; transform labels and mark synthetic occlusion."""
    import numpy as np
    from PIL import ImageDraw
    points=points.copy(); visible=visible.copy()
    size=np.array(image.size,dtype=float)
    valid=(visible>0)&np.isfinite(points).all(axis=1)&(points>=0).all(axis=1)&(points<=1).all(axis=1)
    if valid.any():
        pixels=points[valid]*size
        center=(pixels.min(0)+pixels.max(0))/2
        extent=np.maximum(pixels.max(0)-pixels.min(0),24.)
        extent*=random.uniform(1.5,3.) if augment else 2.
        if augment: center+=np.array([random.uniform(-.12,.12),random.uniform(-.12,.12)])*extent
        low=np.maximum(np.floor(center-extent/2),0).astype(int)
        high=np.minimum(np.ceil(center+extent/2),size).astype(int)
        image=image.crop((*low,*high))
        points=(points*size-low)/(high-low)
    visible*=((points>=0).all(axis=1)&(points<=1).all(axis=1))
    if augment and random.random()<.3:
        # Occlusion labels are changed with the pixels, unlike the legacy
        # unverified visibility flags in near-contact captures.
        width,height=image.size
        x=random.uniform(0.,.65);y=random.uniform(0.,.65)
        w=random.uniform(.2,.5);h=random.uniform(.2,.5)
        ImageDraw.Draw(image).rectangle((x*width,y*height,(x+w)*width,(y+h)*height),
                                       fill=tuple(random.randint(0,100) for _ in range(3)))
        covered=(points[:,0]>=x)&(points[:,0]<=x+w)&(points[:,1]>=y)&(points[:,1]<=y+h)
        visible[covered]=0
    return image,points,visible


def parse_runtime_crop(row):
    """Accept only explicitly prepared, geometry-derived rail crop metadata."""
    record = row.get('runtime_crop')
    if record is None:
        return None
    # Training may also use crops from a GT-derived board for views whose start
    # pose never shows the marker; the runtime always registers from RGB.
    if record.get('preprocessing') != 'rail_crop_v1' or record.get('source') not in {
            'rgb_board_registration', 'privileged_gt_board'}:
        raise ValueError('Unsupported runtime crop provenance')
    digest = record.get('capture_sha256', '')
    if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('Runtime crop requires capture SHA256')
    box = record.get('box_xyxy', [])
    if len(box) != 4 or any(type(value) is not int for value in box):
        raise ValueError('Runtime crop must contain four integer coordinates')
    if box[0] < 0 or box[1] < 0 or box[2]-box[0] < 24 or box[3]-box[1] < 24:
        raise ValueError('Invalid runtime crop bounds')
    return tuple(box)


def rail_crop(image, points, visible, box):
    """Apply exactly the saved runtime window; labels never choose the window."""
    import numpy as np
    if box is None:
        raise ValueError('rail_crop_v1 requires a prepared runtime crop for every sample')
    x0,y0,x1,y1 = box
    if not (0 <= x0 < x1 <= image.width and 0 <= y0 < y1 <= image.height):
        raise ValueError('Runtime crop is outside the captured image')
    mapped = (points*np.array(image.size)-[x0,y0])/[x1-x0,y1-y0]
    visibility = visible.copy()*((mapped>=0).all(axis=1)&(mapped<1).all(axis=1))
    return image.crop(box),mapped,visibility
