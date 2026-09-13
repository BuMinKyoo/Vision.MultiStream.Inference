using System;
using System.Buffers;
using Microsoft.ML.OnnxRuntime.Tensors;
using Vision.MultiStream.Inference.Services.Cuda;

namespace Vision.MultiStream.Inference.Services.Yolo
{
    /// <summary>
    /// YOLOv8 입력 텐서 [1,3,640,640] 만들기. 책임 1개:
    /// "어떤 형태의 픽셀이 들어오든 letterbox + 정규화 + CHW 텐서로 변환".
    /// 진입점: 메모리 BGR byte[] (CPU 경로 / CUDA 커널 경로).
    /// 4단계: (1) letterbox 리사이즈 (2) RGB 정렬 (3) 0~1 정규화 (4) HWC→CHW 차원 재배열.
    /// </summary>
    public static class YoloPreprocessor
    {
        public const int InputSize = 640;

        // YOLOv8 letterbox 표준 패딩 색 (회색)
        private const float PadValueNormalized = 114f / 255f;

        /// <summary>
        /// 메모리상 BGR 픽셀(OpenCV Mat 기본 포맷)을 받아 전처리. (RTSP 프레임 도메인용)
        /// bgrPixels 길이 = width * height * 3, 채널 순서 B-G-R, row-major.
        ///
        /// [Step 6] ImageSharp(이미지 객체 할당 + B↔R 스왑 풀복사 + Resize 재할당)를 제거하고,
        /// BGR byte[] → CHW float 텐서를 단일 unsafe 패스로 만든다:
        ///   (1) letterbox bilinear 리사이즈 (2) B↔R 스왑 (3) 0~1 정규화 (4) HWC→CHW
        /// 텐서 백킹은 ArrayPool 에서 빌려 프레임당 ~4.9MB LOH 할당을 없앤다(추론 후 풀 반납).
        /// 원본 전체(W·H)가 아니라 출력 영역(newW·newH ≈ 640·newH)만 순회하므로 연산량도 줄어든다.
        /// </summary>
        public static unsafe LetterboxResult Preprocess(byte[] bgrPixels, int width, int height)
        {
            const int size = InputSize;            // 640
            int channelStride = size * size;       // 한 채널(640×640) 원소 수
            int tensorLength = 3 * channelStride;  // [1,3,640,640]

            // 종횡비 유지 축소 비율 + letterbox 패딩(중앙 배치).
            float scale = Math.Min((float)size / width, (float)size / height);
            int newW = (int)Math.Round(width * scale);
            int newH = (int)Math.Round(height * scale);
            if (newW < 1) { newW = 1; }
            if (newH < 1) { newH = 1; }
            int padX = (size - newW) / 2;
            int padY = (size - newH) / 2;

            float[] buffer = ArrayPool<float>.Shared.Rent(tensorLength);

            // letterbox 빈 영역(회색)으로 전체 초기화 — Span.Fill 은 SIMD 로 처리(다차원 인덱서 루프보다 빠름).
            // 이후 이미지 영역만 덮어쓰므로 패딩 테두리는 이 값으로 남는다.
            buffer.AsSpan(0, tensorLength).Fill(PadValueNormalized);

            const float inv255 = 1f / 255f;
            int srcStride = width * 3; // BGR

            fixed (byte* srcBase = bgrPixels)
            fixed (float* dstBase = buffer)
            {
                float* rPlane = dstBase;                     // CHW: 채널0 = R
                float* gPlane = dstBase + channelStride;      // 채널1 = G
                float* bPlane = dstBase + 2 * channelStride;  // 채널2 = B

                for (int oy = 0; oy < newH; oy++)
                {
                    // 출력 행 oy → 원본 y (half-pixel center 매핑, bilinear).
                    float srcYf = (oy + 0.5f) / scale - 0.5f;
                    if (srcYf < 0f) { srcYf = 0f; }
                    int y0 = (int)srcYf;
                    if (y0 > height - 1) { y0 = height - 1; }
                    int y1 = y0 + 1 < height ? y0 + 1 : y0;
                    float wy = srcYf - y0;

                    byte* srcRow0 = srcBase + (y0 * srcStride);
                    byte* srcRow1 = srcBase + (y1 * srcStride);
                    int dstRowOffset = ((oy + padY) * size) + padX;

                    for (int ox = 0; ox < newW; ox++)
                    {
                        float srcXf = (ox + 0.5f) / scale - 0.5f;
                        if (srcXf < 0f) { srcXf = 0f; }
                        int x0 = (int)srcXf;
                        if (x0 > width - 1) { x0 = width - 1; }
                        int x1 = x0 + 1 < width ? x0 + 1 : x0;
                        float wx = srcXf - x0;

                        int c0 = x0 * 3; // 좌측 픽셀 BGR 시작
                        int c1 = x1 * 3; // 우측 픽셀 BGR 시작

                        byte* p00 = srcRow0 + c0;
                        byte* p01 = srcRow0 + c1;
                        byte* p10 = srcRow1 + c0;
                        byte* p11 = srcRow1 + c1;

                        // BGR 순서 그대로 bilinear → 정규화하면서 R/G/B 평면에 분배(B↔R 스왑).
                        float b = Bilinear(p00[0], p01[0], p10[0], p11[0], wx, wy);
                        float g = Bilinear(p00[1], p01[1], p10[1], p11[1], wx, wy);
                        float r = Bilinear(p00[2], p01[2], p10[2], p11[2], wx, wy);

                        int di = dstRowOffset + ox;
                        rPlane[di] = r * inv255;
                        gPlane[di] = g * inv255;
                        bPlane[di] = b * inv255;
                    }
                }
            }

            // 풀 버퍼를 그대로 백킹으로 쓰는 텐서(정확 길이로 슬라이스). 추론 후 LetterboxResult.Dispose 가 반납.
            var tensor = new DenseTensor<float>(buffer.AsMemory(0, tensorLength), new[] { 1, 3, size, size });
            return new LetterboxResult(tensor, scale, padX, padY, width, height)
            {
                PooledBuffer = buffer
            };
        }

        /// <summary>
        /// [Phase 4 Step 14] 위 CPU 경로와 동일한 letterbox 결과를 CUDA 커널(vision_cuda.dll)로 만든다.
        /// letterbox 파라미터(scale/newW/newH/padX/padY)는 CPU 버전과 같은 공식으로 여기서 계산해
        /// 네이티브에 넘긴다(반올림 규칙 공유 → 결과 일치). 텐서 백킹은 CPU 경로와 동일하게 ArrayPool 사용.
        /// CUDA 실패(rc!=0) 시 관리 코드 CPU 경로로 폴백해 앱을 계속 돌린다.
        /// </summary>
        public static LetterboxResult PreprocessCuda(byte[] bgrPixels, int width, int height)
        {
            const int size = InputSize;            // 640
            int tensorLength = 3 * size * size;    // [1,3,640,640]

            float scale = Math.Min((float)size / width, (float)size / height);
            int newW = (int)Math.Round(width * scale);
            int newH = (int)Math.Round(height * scale);
            if (newW < 1) { newW = 1; }
            if (newH < 1) { newH = 1; }
            int padX = (size - newW) / 2;
            int padY = (size - newH) / 2;

            float[] buffer = ArrayPool<float>.Shared.Rent(tensorLength);

            int rc = CudaInterop.Preprocess(
                bgrPixels, width, height, newW, newH, padX, padY, scale,
                buffer.AsSpan(0, tensorLength));

            if (rc != 0)
            {
                // CUDA 실패 → 빌린 버퍼 반납하고 CPU 경로로 폴백(원인은 Output 로그로).
                ArrayPool<float>.Shared.Return(buffer);
                System.Diagnostics.Debug.WriteLine($"[YoloPreprocessor] CUDA 전처리 실패(rc={rc}) → CPU 폴백");
                return Preprocess(bgrPixels, width, height);
            }

            var tensor = new DenseTensor<float>(buffer.AsMemory(0, tensorLength), new[] { 1, 3, size, size });
            return new LetterboxResult(tensor, scale, padX, padY, width, height)
            {
                PooledBuffer = buffer
            };
        }

        // 4 이웃(좌상/우상/좌하/우하) 값을 wx, wy 가중으로 bilinear 보간.
        private static float Bilinear(byte v00, byte v01, byte v10, byte v11, float wx, float wy)
        {
            float top = v00 + ((v01 - v00) * wx);
            float bottom = v10 + ((v11 - v10) * wx);
            return top + ((bottom - top) * wy);
        }
    }
}
