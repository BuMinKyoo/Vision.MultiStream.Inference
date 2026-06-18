# -*- coding: utf-8 -*-
"""FP32 vs INT8 출력 비교 (양자화 정상 여부 빠른 점검)."""
import os
import glob
import numpy as np
import onnxruntime as ort

from quantize_int8 import preprocess, MODEL_IN, MODEL_OUT, FRAMES_DIR

frames = sorted(glob.glob(os.path.join(FRAMES_DIR, "*.jpg")))[:5]
s32 = ort.InferenceSession(MODEL_IN, providers=["CPUExecutionProvider"])
s8 = ort.InferenceSession(MODEL_OUT, providers=["CPUExecutionProvider"])
name = s32.get_inputs()[0].name

for f in frames:
    x = preprocess(f).astype(np.float32)
    o32 = s32.run(None, {name: x})[0][0]   # [84,8400]
    o8 = s8.run(None, {name: x})[0][0]
    c32 = float(o32[4:].max())             # 클래스 점수 최대 = 검출 강도
    c8 = float(o8[4:].max())
    a, b = o32.ravel(), o8.ravel()
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    print(f"{os.path.basename(f)}: maxscore fp32={c32:.3f} int8={c8:.3f}  cos-sim={cos:.4f}")
