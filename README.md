# A Toy demo of FingerPrint

A toy demo for fingerprint of python implement, and theory 
is based on shazam`s method. 

It`s very tiny code(less than ~150 lines) but has robust recognized result.

## Usage
```python
import librosa
import os
from fingerprint import FpAnchor, FpEngine

hp = {
"hop_size": 1024,
"hop_length": 2048,
"map_margin": 100,
"frame_min": 10,
"amp_min": 5,
"fp_freq_margin_min": 5
} # recommend params

sr = 8000
fp_anchor = FpAnchor(hp)
wav_dir = "debug_wav"
for wav_name in os.listdir(wav_dir):
    wav_path = os.path.join(wav_dir, wav_name)
    signal, _ = librosa.load(wav_path, sr=sr)
    fp_anchor.add_anchor(signal, wav_name.split(".")[0])

engine = FpEngine(hp, fp_anchor=fp_anchor)
wav_path = os.path.join(wav_dir, os.listdir(wav_dir)[0])
target_sig, _ = librosa.load(wav_path, sr=sr)
target_sig = target_sig[:sr * 15]  # use one anchor(first 15s) audio as target
res = engine.recognize(target_sig, topk=1)
print("top5:", res[0])
```

## Reference

- how-shazam-works: http://coding-geek.com/how-shazam-works/
