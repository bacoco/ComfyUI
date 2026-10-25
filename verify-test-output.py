"""Decode the generated MP4 and record its actual video and audio properties."""
import json
import pathlib

import av
import numpy as np

root = pathlib.Path(__file__).resolve().parent
result = json.loads((root / 'test-evidence/gpu-test-result.json').read_text())
prompt = json.loads((root / 'user/default/fasth3-test-api.json').read_text())
expected = prompt['7']['inputs']
assert result['history']['status']['status_str'] == 'success', 'Generation failed'
assert result['original_service_restored'], 'Original ComfyUI not restored'
prefix = pathlib.Path(prompt['12']['inputs']['filename_prefix'])
files = sorted((root / 'output' / prefix.parent).glob(prefix.name + '_*.mp4'))
assert files, 'Missing MP4'
path = files[-1]
with av.open(str(path)) as container:
    stream = container.streams.video[0]
    video = {'width': stream.width, 'height': stream.height,
             'fps': float(stream.average_rate), 'codec': stream.codec_context.name}
    count = 0
    for frame in container.decode(video=0):
        if count in (0, expected['length'] // 2, expected['length'] - 1):
            frame.to_image().save(root / f'test-evidence/frame-{count:03d}.png')
        count += 1
    video['frames'] = count
with av.open(str(path)) as container:
    stream = container.streams.audio[0]
    audio = {'sample_rate': stream.sample_rate, 'channels': stream.codec_context.channels,
             'codec': stream.codec_context.name}
    samples = 0
    energy = 0.0
    elements = 0
    for frame in container.decode(audio=0):
        values = frame.to_ndarray().astype(np.float64)
        if frame.format.is_planar:
            samples += values.shape[1]
        else:
            samples += frame.samples
        energy += float(np.square(values).sum())
        elements += values.size
    audio['duration_seconds'] = samples / audio['sample_rate']
    audio['rms'] = (energy / elements) ** 0.5
assert video['width'] == expected['width'] and video['height'] == expected['height']
assert video['frames'] == expected['length'] and video['fps'] == 24
assert audio['channels'] == 2 and audio['rms'] > 0
evidence = {'file': str(path), 'video': video, 'audio': audio,
            'elapsed_seconds_first_request': result['elapsed_seconds'],
            'original_service_restored': result['original_service_restored']}
(root / 'test-evidence/media-verification.json').write_text(json.dumps(evidence, indent=2))
print(json.dumps(evidence, indent=2))
