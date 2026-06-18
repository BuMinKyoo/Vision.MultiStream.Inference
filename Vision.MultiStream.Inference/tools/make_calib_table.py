# -*- coding: utf-8 -*-
"""
Approach A: TensorRT INT8 용 캘리브레이션 테이블 생성.

QDQ 모델(B)이 구버전 TRT(ORT 1.16.2)에서 Q/DQ shape inference 로 빌드 실패하므로,
모델은 잘 빌드되는 FP32 yolov8n.onnx 를 그대로 두고, TRT 가 내부에서 INT8 변환할 때 쓸
활성값 dynamic range 만 외부 테이블(calibration.flatbuffers)로 만들어 둔다.

흐름:
  1) (재)추출된 프레임을 FP32 모델에 통과시키며 각 텐서의 min/max 분포 수집.
  2) calibration.flatbuffers / .cache / .json 을 Assets/Models 에 기록.
  3) C# TRT EP 에서 trt_int8_enable=1 + trt_int8_calibration_table_name 으로 사용.

실행:
  python tools/make_calib_table.py
"""
import os
import glob

from onnxruntime.quantization import create_calibrator, CalibrationMethod, write_calibration_table

from quantize_int8 import (
    MODEL_IN, MODEL_DIR, FRAMES_DIR, FrameReader, extract_frames, MAX_FRAMES,
)

AUG_MODEL = os.path.join(MODEL_DIR, "augmented_calib.onnx")  # 캘리브레이션용 임시 증강 모델


def main():
    # 이미 추출된 프레임이 있으면 재사용, 없으면 영상에서 추출.
    frames = sorted(glob.glob(os.path.join(FRAMES_DIR, "*.jpg")))
    if frames:
        if len(frames) > MAX_FRAMES:
            frames = frames[:MAX_FRAMES]
        print(f"[1/3] 기존 추출 프레임 재사용: {len(frames)}장")
    else:
        frames = extract_frames()

    print("[2/3] FP32 모델로 캘리브레이션(min/max 수집)")
    calibrator = create_calibrator(
        MODEL_IN,
        op_types_to_calibrate=None,                 # 기본 대상 모두
        augmented_model_path=AUG_MODEL,
        calibrate_method=CalibrationMethod.MinMax,  # TRT 대칭 INT8 에 적합
    )
    calibrator.set_execution_providers(["CPUExecutionProvider"])
    calibrator.collect_data(FrameReader(frames))
    ranges = calibrator.compute_data()

    print("[3/3] 캘리브레이션 테이블 기록 -> calibration.flatbuffers")
    write_calibration_table(ranges, dir=MODEL_DIR)

    if os.path.isfile(AUG_MODEL):
        os.remove(AUG_MODEL)

    for f in ("calibration.flatbuffers", "calibration.cache", "calibration.json"):
        p = os.path.join(MODEL_DIR, f)
        if os.path.isfile(p):
            print(f"  생성됨: {p}  ({os.path.getsize(p)} bytes)")


if __name__ == "__main__":
    main()
