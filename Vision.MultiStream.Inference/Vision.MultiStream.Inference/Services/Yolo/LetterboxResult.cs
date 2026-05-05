using Microsoft.ML.OnnxRuntime.Tensors;

namespace Vision.MultiStream.Inference.Services.Yolo
{
    /// <summary>
    /// 전처리 결과. 추론 후 letterbox 좌표를 원본으로 되돌리는 데
    /// Scale/PadX/PadY/Original* 가 사용됨.
    /// </summary>
    public sealed record LetterboxResult(
        DenseTensor<float> Tensor, // ONNX 입력 텐서 [1, 3, 640, 640] (CHW, 0~1 정규화)
        float Scale,               // 원본 → 640 리사이즈 비율 (역변환 시 나눔)
        int PadX,                  // 좌우 패딩 픽셀 수 (역변환 시 뺌)
        int PadY,                  // 상하 패딩 픽셀 수 (역변환 시 뺌)
        int OriginalWidth,         // 원본 이미지 너비 (박스 클램프 범위)
        int OriginalHeight);       // 원본 이미지 높이 (박스 클램프 범위)
}
