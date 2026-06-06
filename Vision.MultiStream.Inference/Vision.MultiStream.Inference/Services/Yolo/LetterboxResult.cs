using System;
using System.Buffers;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Vision.MultiStream.Inference.Services.Yolo
{
    /// <summary>
    /// 전처리 결과. 추론 후 letterbox 좌표를 원본으로 되돌리는 데
    /// Scale/PadX/PadY/Original* 가 사용됨.
    ///
    /// 텐서 백킹 버퍼가 ArrayPool 에서 빌려온 것이면 <see cref="PooledBuffer"/> 에 보관되며,
    /// 추론이 끝난 뒤 <see cref="Dispose"/> 로 풀에 반납한다(프레임당 LOH 할당 제거).
    /// 파일 경로(스냅샷)처럼 풀을 쓰지 않는 경우 PooledBuffer 는 null 이고 Dispose 는 no-op.
    /// </summary>
    public sealed record LetterboxResult(
        DenseTensor<float> Tensor, // ONNX 입력 텐서 [1, 3, 640, 640] (CHW, 0~1 정규화)
        float Scale,               // 원본 → 640 리사이즈 비율 (역변환 시 나눔)
        int PadX,                  // 좌우 패딩 픽셀 수 (역변환 시 뺌)
        int PadY,                  // 상하 패딩 픽셀 수 (역변환 시 뺌)
        int OriginalWidth,         // 원본 이미지 너비 (박스 클램프 범위)
        int OriginalHeight)        // 원본 이미지 높이 (박스 클램프 범위)
        : IDisposable
    {
        // 풀에서 빌린 텐서 백킹 버퍼(없으면 null). Dispose 시 풀로 반납.
        internal float[]? PooledBuffer { get; init; }

        public void Dispose()
        {
            float[]? pooled = PooledBuffer;
            if (pooled != null)
            {
                // clearArray=false: 다음 Rent 사용자가 전체를 Fill 후 사용하므로 초기화 불필요.
                ArrayPool<float>.Shared.Return(pooled);
            }
        }
    }
}
