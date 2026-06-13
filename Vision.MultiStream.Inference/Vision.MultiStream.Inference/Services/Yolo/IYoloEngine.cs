using System.Collections.Generic;
using System.Threading;
using Vision.MultiStream.Inference.Models;

namespace Vision.MultiStream.Inference.Services.Yolo
{
    /// <summary>
    /// YOLO 추론 엔진 추상화. 전처리 결과(LetterboxResult)를 받아 검출 리스트를 돌려준다.
    /// 구현체: 관리 코드 ONNX Runtime(<see cref="YoloInferenceEngine"/>),
    /// 네이티브 DLL "GPU(C++)"(<see cref="NativeYoloEngine"/>).
    /// 전처리(YoloPreprocessor)와 후처리(YoloPostprocessor)는 구현체가 공유한다.
    /// </summary>
    public interface IYoloEngine
    {
        InferenceDevice Device { get; }

        (IReadOnlyList<Detection> Detections, double InferenceMs, double PostprocessMs) Detect(
            LetterboxResult lb, CancellationToken cancellationToken = default);
    }
}
