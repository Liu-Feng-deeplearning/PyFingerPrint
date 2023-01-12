# A Toy demo of FingerPrint

A toy demo for fingerprint of python implement, and theory 
is based on shazam`s method. 

It`s very tiny code(less than ~150 lines) but has robust recognized result.

## Finger Details
Core perception of shazam's method is landmark and finger. Landmark is point 
on feature-map which has higher magnitude than the surrounding points. 
We also call landmark as "peaks" in python program.
And finger is turple contains two landmark's position of freq-axis 
and their distance of time-axis.
We save finger as "0|100|10", it means this finger contains two landmark at (x, 0) and (x+10, 100).
And x is not recorded because we don't care about landmark's absolute position in time-axis.
Of course, we can use a hash function to map "0|100|10" into string or int with fixed length(e.g. 16 bits).
        
After lots of experiment, we found some tricks about signal and feature to speed up computing pipeline 
and decrease num of fingers. Sample rate is 8k enough, and hop-size/hop-length are recommended as 1024 and 2048.
Other important hparams are written at Usage's toy demo. 
  
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
# we register for songs as anchors 
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
