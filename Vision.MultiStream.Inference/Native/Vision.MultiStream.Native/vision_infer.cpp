// vision_infer.cpp
//
// 마일스톤 2 구현: ONNX Runtime C++ API 로 실제 추론.
// 경계는 "추론 엔진만" — 전처리([1,3,640,640] float)는 C# YoloPreprocessor 가,
// 후처리(ParseOutput/NMS)는 C# 가 담당한다. 이 DLL 은:
//   vinfer_create : 모델 로드 + DirectML EP 세션 구성
//   vinfer_infer  : 호출자가 핀한 input 텐서를 받아 Run, raw 출력을 호출자 output 버퍼에 기록
// 입력/출력 모두 호출자(C#) 버퍼를 CPU 메모리 OrtValue 로 감싸 ORT 에 직접 넘기므로
// 네이티브 쪽 프레임당 신규 할당이 없다(C# 의 IoBinding 출력 재사용과 동일 철학).

#include "vision_infer.h"

#include <onnxruntime_cxx_api.h>
#include <dml_provider_factory.h>

#include <array>
#include <mutex>
#include <string>
#include <vector>

namespace {

// 핸들 뒤의 추론 컨텍스트. 세션/이름/형상/메모리정보를 보유.
struct VinferContext
{
    Ort::Env                 env{ORT_LOGGING_LEVEL_WARNING, "vinfer"};
    Ort::Session             session{nullptr};
    Ort::MemoryInfo          memInfo{nullptr};

    std::string              inputName;
    std::string              outputName;
    std::array<const char*, 1> inputNames{};   // session.Run 용 (c_str 캐시)
    std::array<const char*, 1> outputNames{};

    std::vector<int64_t>     inputShape;        // [1,3,640,640]
    std::vector<int64_t>     outputShape;       // [1,84,8400] 등
    size_t                   inputLen  = 0;
    size_t                   outputLen = 0;

    std::mutex               mtx;               // 세션 동시 호출 보호(방어적)
};

// 동적 차원(-1)이면 고정값으로 대체. YOLOv8 고정 export 가정과 정합.
void fixDims(std::vector<int64_t>& dims, const std::vector<int64_t>& fallback)
{
    if (dims.size() != fallback.size())
    {
        dims = fallback;
        return;
    }
    for (size_t i = 0; i < dims.size(); ++i)
    {
        if (dims[i] <= 0)
        {
            dims[i] = fallback[i];
        }
    }
}

size_t elemCount(const std::vector<int64_t>& dims)
{
    size_t n = 1;
    for (int64_t d : dims)
    {
        n *= static_cast<size_t>(d > 0 ? d : 0);
    }
    return n;
}

} // namespace

extern "C" {

VINFER_API VinferHandle vinfer_create(const wchar_t* model_path, int device)
{
    if (model_path == nullptr)
    {
        return nullptr;
    }

    VinferContext* ctx = nullptr;
    try
    {
        ctx = new VinferContext();

        Ort::SessionOptions opts;
        // DirectML EP 권장 설정: 메모리 패턴 비활성 + 순차 실행.
        opts.DisableMemPattern();
        opts.SetExecutionMode(ORT_SEQUENTIAL);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        if (device == 1) // 1 = DirectML
        {
            Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DML(opts, 0));
        }
        // device == 0 (CPU): EP 추가 없이 기본 CPU 실행.

        ctx->session = Ort::Session(ctx->env, model_path, opts);

        // 입력/출력 이름·형상 메타데이터 추출.
        Ort::AllocatorWithDefaultOptions alloc;

        auto inNameAlloc  = ctx->session.GetInputNameAllocated(0, alloc);
        auto outNameAlloc = ctx->session.GetOutputNameAllocated(0, alloc);
        ctx->inputName  = inNameAlloc.get();
        ctx->outputName = outNameAlloc.get();
        ctx->inputNames[0]  = ctx->inputName.c_str();
        ctx->outputNames[0] = ctx->outputName.c_str();

        ctx->inputShape  = ctx->session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        ctx->outputShape = ctx->session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

        // YOLOv8: 입력 [1,3,640,640] 고정. 동적이면 이 값으로 보정.
        fixDims(ctx->inputShape, {1, 3, 640, 640});
        // 출력은 모델 메타데이터 형상을 신뢰(예 [1,84,8400]). 동적 batch 만 1 로.
        if (!ctx->outputShape.empty() && ctx->outputShape[0] <= 0)
        {
            ctx->outputShape[0] = 1;
        }

        ctx->inputLen  = elemCount(ctx->inputShape);
        ctx->outputLen = elemCount(ctx->outputShape);

        ctx->memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        return ctx;
    }
    catch (const std::exception&)
    {
        delete ctx;
        return nullptr;
    }
}

VINFER_API int vinfer_infer(VinferHandle handle, const float* input, float* output, int output_len)
{
    auto* ctx = static_cast<VinferContext*>(handle);
    if (ctx == nullptr)
    {
        return -1;
    }
    if (input == nullptr || output == nullptr || output_len <= 0)
    {
        return -2;
    }
    if (static_cast<size_t>(output_len) != ctx->outputLen)
    {
        return -3; // 호출자 출력 버퍼 길이가 모델 출력과 불일치
    }

    try
    {
        std::lock_guard<std::mutex> lock(ctx->mtx);

        // 호출자 버퍼를 그대로 가리키는 입력/출력 텐서(복사 없음).
        // ORT 가 출력 버퍼에 결과를 in-place 로 채운다(DirectML 은 device→이 CPU 버퍼로 다운로드).
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            ctx->memInfo, const_cast<float*>(input), ctx->inputLen,
            ctx->inputShape.data(), ctx->inputShape.size());

        Ort::Value outputTensor = Ort::Value::CreateTensor<float>(
            ctx->memInfo, output, ctx->outputLen,
            ctx->outputShape.data(), ctx->outputShape.size());

        ctx->session.Run(
            Ort::RunOptions{nullptr},
            ctx->inputNames.data(), &inputTensor, 1,
            ctx->outputNames.data(), &outputTensor, 1);

        return 0;
    }
    catch (const std::exception&)
    {
        return -10; // ORT 실행 중 예외
    }
}

VINFER_API int vinfer_output_dims(VinferHandle handle, int* num_channels, int* num_anchors)
{
    auto* ctx = static_cast<VinferContext*>(handle);
    if (ctx == nullptr || num_channels == nullptr || num_anchors == nullptr)
    {
        return -1;
    }
    if (ctx->outputShape.size() < 3)
    {
        return -2; // [1,C,A] 3차원 출력 가정
    }
    *num_channels = static_cast<int>(ctx->outputShape[1]);
    *num_anchors  = static_cast<int>(ctx->outputShape[2]);
    return 0;
}

VINFER_API void vinfer_destroy(VinferHandle handle)
{
    delete static_cast<VinferContext*>(handle);
}

} // extern "C"
