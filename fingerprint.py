#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author:liufeng
# datetime:2021/11/19 2:16 PM
# software: PyCharm

"""Python Implement of finger-print as shazam:

http://coding-geek.com/how-shazam-works/

"""

from itertools import groupby

import numpy as np
from scipy.fftpack import fft


class FpExtractor:
  """Extractor of fingerprint, computer finger print with signal.
  Input: signal (recommend using sample-rate of 8k)
  Output: fingers
  """

  def __init__(self, hp):
    self._hp = hp
    return

  def sig_to_fp(self, signal):
    """signal to fingerprint"""
    spec = self._sig_to_spec(signal)
    peaks = self._spec_to_peaks(spec)
    return self._peaks_to_fp(peaks)

  def _sig_to_spec(self, signal):
    """signal -> spec """
    win_shift, win_len = self._hp["hop_size"], self._hp["hop_length"]
    time_length = (len(signal) - win_len) // win_shift + 1
    spec = np.zeros((time_length, win_len // 2))
    for idx in range(time_length):
      start = idx * win_shift
      dst = fft(signal[start:start + win_len])[0:win_len // 2]
      spec[idx] = np.abs(dst)
    return spec.T

  def _spec_to_peaks(self, spec):
    """spec -> peaks(local max of spec)"""
    max_freq, time_size = np.shape(spec)
    peaks = []
    for time_idx in range(time_size):
      local_freq = spec[:, time_idx]
      local_xy = [(freq_idx, time_idx)
                  for freq_idx in self._spec1d_to_peaks(local_freq)]
      peaks.extend(local_xy)
    return peaks

  def _spec1d_to_peaks(self, spec_1d):
    """spec of 1d -> peaks(local max of spec)"""
    local_max = []
    max_freq = np.size(spec_1d)
    margin = 25
    start = 0
    end = margin * 2
    while start < max_freq:
      if len(local_max) >= 2:
        margin = self._hp["map_margin"]
      idx = np.argmax(spec_1d[start:end])
      idx = idx + start
      amp = 10 * np.log10(spec_1d[idx] + 1e-3)
      if amp > self._hp["amp_min"]:
        local_max.append(idx)
      start = idx + margin
      end = start + margin * 2
    return local_max

  def _peaks_to_fp(self, peaks, min_time_delta=0, max_time_delta=1000):
    """ extract fp data from peaks

    Args:
      peaks: list of peak frequencies and times.
      min_time_delta: min frame-wise distance to which a peak can be paired
                      with its neighbors.
      max_time_delta: max frame-wise distance to which a peak can be paired
                      with its neighbors.

    Returns:
      fp_data: (hash_id, frame)
    """
    peaks = sorted(peaks, key=lambda x: (x[1], x[0]))
    fp_data = []
    for i in range(len(peaks)):
      j = 1
      while (i + j) < len(peaks):
        freq1, t1 = peaks[i]
        freq2, t2 = peaks[i + j]
        t_delta = t2 - t1
        if t_delta < min_time_delta:
          pass
        elif t_delta > max_time_delta:
          break
        else:
          if abs(freq2 - freq1) >= self._hp["fp_freq_margin_min"]:
            fp_data.append((f"{str(freq1)}|{str(freq2)}|{str(t_delta)}", t1))
        j += 1
        if j == self._hp["frame_min"]:
          break
    return fp_data


class FpAnchor:
  """Process for all signal"""

  def __init__(self, hp):
    self._hp = hp
    self._anchor_set = set()
    self._anchor_fp = {}
    self._fp_extractor = FpExtractor(hp)
    return

  def add_anchor(self, signal, anchor_name):
    """add signal into engine, and stored song name as anchor name"""
    finger_print = self._fp_extractor.sig_to_fp(signal)
    self._anchor_set.add(anchor_name)
    for hash_key, t1 in finger_print:
      if hash_key not in self._anchor_fp.keys():
        self._anchor_fp[hash_key] = []
      self._anchor_fp[hash_key].append((anchor_name, t1))
    return

  def get_anchor_fp(self):
    return self._anchor_fp

  def get_fp_extractor(self):
    return self._fp_extractor


class FpEngine:
  """Engine of fingerprint, needed to add FpAnchor before recognizing"""

  def __init__(self, hp, fp_anchor: FpAnchor):
    self._hp = hp
    self._fp_extractor = fp_anchor.get_fp_extractor()
    self._fp_anchor = fp_anchor.get_anchor_fp()
    return

  def recognize(self, signal, topk=5):
    finger_prints = self._fp_extractor.sig_to_fp(signal)
    matches_fp = []

    for hashes, now_frame in finger_prints:
      if hashes in self._fp_anchor.keys():
        for h in self._fp_anchor[hashes]:
          matches_fp.append((h[0], h[1] - now_frame, now_frame))

    matches_fp = sorted(matches_fp, key=lambda m: (m[0], m[1]))
    counts = [(*key, len(list(group))) for key, group in
              groupby(matches_fp,
                      key=lambda m: (m[0], m[1]))]  # merge pairs with same diff

    songs_matches = [max(list(group), key=lambda g: g[2]) for key, group in
                     groupby(counts, key=lambda count: count[
                       0])]  # get max line with k=1 for every song

    songs_matches = sorted(songs_matches, key=lambda _x: _x[2], reverse=True)
    result_topk = []
    for x in songs_matches[0: topk]:
      res = {"name": x[0], "count": x[2], "frame": x[1]}
      result_topk.append(res)
    return result_topk


def _test():
  import librosa
  import os
  hp = {
    "hop_size": 1024,
    "hop_length": 2048,
    "map_margin": 100,
    "frame_min": 10,
    "amp_min": 5,
    "fp_freq_margin_min": 5
  }

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
  res = engine.recognize(target_sig)
  print("top5:", res[0])
  return


if __name__ == '__main__':
  _test()