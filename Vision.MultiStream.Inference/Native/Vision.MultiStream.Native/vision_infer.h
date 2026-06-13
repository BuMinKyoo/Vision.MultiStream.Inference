// vision_infer.h
//
// Vision.MultiStream.Inference 의 네이티브 추론 엔진(Phase 3 "GPU(C++)") C ABI.
// C# 의 GPU(C++) 라디오 경로가 P/Invoke 로 호출한다. extern "C" 평면 ABI 로
// 이름 데코레이션 없이 export 하여 DllImport 매핑을 단순하게 유지한다.
//
// 경계 설계(마일스톤 1~3):
//   - 전처리([1,3,640,640] float 텐서)와 후처리(ParseOutput/NMS)는 C# 가 그대로 담당.
//   - 이 DLL 은 "추론 엔진만" 책임진다: 입력 텐서를 받아 ORT 세션을 돌리고
//     raw 출력([1,84,8400] 등)을 호출자가 제공한 버퍼에 in-place 로 기록.
//   - 입력/출력 모두 호출자(C#)가 핀(fixed)한 버퍼를 가리키는 포인터 → 복사 없음.
//
// 마일스톤 1: 함수 시그니처와 핸들 수명만 검증(추론은 no-op).

#pragma once

#include <cstdint>

#ifdef VISION_INFER_EXPORTS
#define VINFER_API __declspec(dllexport)
#else
#define VINFER_API __declspec(dllimport)
#endif

// 불투명 핸들. 내부적으로 ORT 세션/바인딩/버퍼를 소유하는 컨텍스트를 가리킨다.
typedef void* VinferHandle;

#ifdef __cplusplus
extern "C" {
#endif

// 모델을 로드하고 추론 컨텍스트를 만든다.
//   model_path : ONNX 모델 절대 경로(UTF-16, Windows 와이드 문자열)
//   device     : 0=CPU, 1=DirectML (Phase 3 는 DirectML 유지)
// 반환: 성공 시 핸들, 실패 시 nullptr.
VINFER_API VinferHandle vinfer_create(const wchar_t* model_path, int device);

// 단일 추론. input/output 은 호출자가 핀한 버퍼.
//   input      : [1,3,640,640] = 1*3*640*640 float (CHW, 정규화 완료)
//   output     : raw 출력 버퍼 (예: [1,84,8400]). output_len = 원소 수.
// 반환: 0=성공, 음수=오류.
VINFER_API int vinfer_infer(VinferHandle handle, const float* input, float* output, int output_len);

// 모델 출력 형상([1, num_channels, num_anchors], 예 [1,84,8400])을 돌려준다.
// C# 가 출력 버퍼 길이(num_channels*num_anchors)와 후처리 스트라이드를 하드코딩 없이 알기 위함.
// 반환: 0=성공, 음수=오류.
VINFER_API int vinfer_output_dims(VinferHandle handle, int* num_channels, int* num_anchors);

// 컨텍스트와 모든 네이티브 리소스를 해제한다. handle 이 nullptr 이면 무시.
VINFER_API void vinfer_destroy(VinferHandle handle);

#ifdef __cplusplus
}
#endif
