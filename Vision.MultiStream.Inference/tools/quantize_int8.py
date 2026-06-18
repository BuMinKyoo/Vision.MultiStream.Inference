# -*- coding: utf-8 -*-
"""
YOLOv8 ONNX(FP32) -> INT8 QDQ 모델 생성 스크립트 (B 방식).

흐름:
  1) 추적 대상 영상들(VIDEO_DIR)에서 ffmpeg 로 프레임을 고르게 추출 (영상당 FRAMES_PER_VIDEO 장).
  2) 추출 프레임을 C# 추론(YoloPreprocessor)과 동일하게 전처리:
     레터박스 640x640(종횡비 유지, 중앙 배치, 패딩색 114) + RGB + 0~1 정규화 + NCHW.
  3) 이 프레임들로 활성값 분포를 관찰(캘리브레이션)하여 INT8 scale 산출.
  4) Q/DQ 노드를 모델에 삽입한 yolov8n_int8.onnx 저장 (TensorRT INT8 용, 대칭 양자화).

실행:
  python tools/quantize_int8.py
"""
import os
import sys
import glob
import shutil
import random
import subprocess

import numpy as np
import onnx
from PIL import Image
from onnxruntime.quantization import (
    quantize_static, CalibrationDataReader, QuantType, QuantFormat, CalibrationMethod,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

# ===== 경로/설정 =====
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "..", "Vision.MultiStream.Inference", "Assets", "Models")
MODEL_IN = os.path.normpath(os.path.join(MODEL_DIR, "yolov8n.onnx"))          # FP32 원본
MODEL_PRE = os.path.normpath(os.path.join(MODEL_DIR, "yolov8n.preproc.onnx")) # 전처리(shape infer) 임시본
MODEL_OUT = os.path.normpath(os.path.join(MODEL_DIR, "yolov8n_int8.onnx"))    # 산출물

# 주의: 이 QDQ(B) 모델은 CPU/DML INT8 에는 쓸 수 있으나, 구버전 TRT(ORT 1.16.2)에서는
# Q/DQ shape inference 로 엔진 빌드가 실패한다. TensorRT INT8 은 make_calib_table.py(A 방식) 사용.

VIDEO_DIR = r"D:\bmkfile\myfile\영상테스트\bin"
FFMPEG = os.path.join(VIDEO_DIR, "ffmpeg.exe")
FFPROBE = os.path.join(VIDEO_DIR, "ffprobe.exe")
FRAMES_DIR = os.path.join(HERE, "_calib_frames")

INPUT_NAME = "images"     # 모델 입력 텐서 이름
SIZE = 640                # 입력 한 변
PAD = 114.0               # 레터박스 패딩색 (YOLOv8 표준)
FRAMES_PER_VIDEO = 16     # 영상당 추출 프레임 수
MAX_FRAMES = 256          # 캘리브레이션에 쓸 최대 프레임 수(많으면 느려짐)


def probe_duration(path):
    """ffprobe 로 영상 길이(초)를 구한다. 실패 시 None."""
    try:
        out = subprocess.run(
            [FFPROBE, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, check=True,
        )
        return float(out.stdout.strip())
    except Exception as e:
        print(f"  [warn] ffprobe 실패({os.path.basename(path)}): {e}")
        return None


def extract_frames():
    """모든 mp4 에서 프레임을 고르게 뽑아 FRAMES_DIR 에 jpg 로 저장."""
    if os.path.isdir(FRAMES_DIR):
        shutil.rmtree(FRAMES_DIR)
    os.makedirs(FRAMES_DIR, exist_ok=True)

    videos = sorted(glob.glob(os.path.join(VIDEO_DIR, "*.mp4")))
    if not videos:
        sys.exit(f"[error] 영상이 없습니다: {VIDEO_DIR}")

    print(f"[1/4] 프레임 추출: {len(videos)}개 영상 -> {FRAMES_DIR}")
    for idx, v in enumerate(videos):
        dur = probe_duration(v)
        # 영상 전체에 고르게 퍼지도록 fps = 프레임수/길이. 길이 모르면 5초 간격.
        if dur and dur > 0:
            fps = max(FRAMES_PER_VIDEO / dur, 0.01)
        else:
            fps = 1.0 / 5.0
        prefix = os.path.join(FRAMES_DIR, f"v{idx:02d}_%03d.jpg")
        try:
            subprocess.run(
                [FFMPEG, "-hide_banner", "-loglevel", "error", "-i", v,
                 "-vf", f"fps={fps:.6f}", "-frames:v", str(FRAMES_PER_VIDEO),
                 "-q:v", "3", prefix],
                check=True,
            )
        except Exception as e:
            print(f"  [warn] 추출 실패({os.path.basename(v)}): {e}")
        got = len(glob.glob(os.path.join(FRAMES_DIR, f"v{idx:02d}_*.jpg")))
        print(f"  - {os.path.basename(v)}: {got} 프레임")

    frames = sorted(glob.glob(os.path.join(FRAMES_DIR, "*.jpg")))
    if not frames:
        sys.exit("[error] 추출된 프레임이 없습니다.")
    if len(frames) > MAX_FRAMES:
        random.seed(0)
        frames = random.sample(frames, MAX_FRAMES)
    print(f"  => 캘리브레이션 사용 프레임: {len(frames)}장")
    return frames


def preprocess(path):
    """C# YoloPreprocessor 와 동일: 레터박스 + RGB + /255 + NCHW float32."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = min(SIZE / w, SIZE / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    img = img.resize((nw, nh), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)          # HWC, RGB, 0~255

    canvas = np.full((SIZE, SIZE, 3), PAD, dtype=np.float32)  # 회색 패딩
    px, py = (SIZE - nw) // 2, (SIZE - nh) // 2
    canvas[py:py + nh, px:px + nw, :] = arr
    canvas /= 255.0                                   # 0~1 정규화
    chw = np.transpose(canvas, (2, 0, 1))            # HWC -> CHW
    return chw[np.newaxis, ...].copy()               # [1,3,640,640]


class FrameReader(CalibrationDataReader):
    """추출 프레임을 한 장씩 모델 입력으로 흘려보내는 캘리브레이션 데이터 리더."""
    def __init__(self, files):
        self.files = files
        self.i = 0

    def get_next(self):
        if self.i >= len(self.files):
            return None
        f = self.files[self.i]
        self.i += 1
        return {INPUT_NAME: preprocess(f)}

    def rewind(self):
        self.i = 0


def prune_unused_opsets(model_path):
    """노드가 실제로 사용하는 도메인의 opset import 만 남기고 나머지는 제거."""
    m = onnx.load(model_path)
    used = set(n.domain for n in m.graph.node)  # "" = ai.onnx 기본
    keep = [op for op in m.opset_import if op.domain in used]
    del m.opset_import[:]
    m.opset_import.extend(keep)
    onnx.save(m, model_path)
    print(f"  opset import 정리: {[ (op.domain or 'ai.onnx', op.version) for op in keep ]}")


def main():
    if not os.path.isfile(MODEL_IN):
        sys.exit(f"[error] 원본 모델 없음: {MODEL_IN}")

    frames = extract_frames()

    print("[2/4] 모델 전처리(shape inference)")
    quant_pre_process(MODEL_IN, MODEL_PRE, skip_symbolic_shape=False)

    # 검출 헤드(/model.22/) 제외: 이 헤드의 마지막 Concat 이 박스좌표(값~640)와 클래스
    # 점수(0~1)를 한 텐서로 합치는데, 양자화하면 단일 scale 이 큰 박스좌표에 맞춰져
    # 점수가 0 으로 뭉개진다. 헤드는 연산량이 작아 제외해도 속도 이득은 거의 유지된다.
    head = onnx.load(MODEL_PRE)
    exclude = [n.name for n in head.graph.node if n.name and "/model.22/" in n.name]
    print(f"  양자화 제외 노드(검출 헤드): {len(exclude)}개")

    print("[3/4] 캘리브레이션 + INT8 양자화 (QDQ, 대칭, per-channel)")
    quantize_static(
        MODEL_PRE,
        MODEL_OUT,
        calibration_data_reader=FrameReader(frames),
        quant_format=QuantFormat.QDQ,        # TensorRT 가 읽는 Q/DQ 노드 삽입
        per_channel=True,                    # 가중치 채널별 scale (정확도↑)
        activation_type=QuantType.QInt8,     # TRT INT8 은 부호있는 8비트
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        nodes_to_exclude=exclude,            # 검출 헤드는 FP 유지 (점수 보존)
        extra_options={                      # TensorRT 는 대칭 양자화 필요
            "ActivationSymmetric": True,
            "WeightSymmetric": True,
        },
    )

    if os.path.isfile(MODEL_PRE):
        os.remove(MODEL_PRE)

    # 양자화 전처리가 끼워 넣은, 실제로는 쓰지 않는 opset import 제거.
    # (ai.onnx.ml=5 등이 남아 있으면 구버전 ORT(앱의 1.16.2)가 로드 거부함)
    prune_unused_opsets(MODEL_OUT)

    sz_in = os.path.getsize(MODEL_IN) / 1024 / 1024
    sz_out = os.path.getsize(MODEL_OUT) / 1024 / 1024
    print("[4/4] 완료")
    print(f"  원본 : {MODEL_IN}  ({sz_in:.1f} MB)")
    print(f"  INT8 : {MODEL_OUT}  ({sz_out:.1f} MB)")


if __name__ == "__main__":
    main()
