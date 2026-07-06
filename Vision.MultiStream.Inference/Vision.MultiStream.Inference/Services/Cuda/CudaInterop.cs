using System;
using System.Runtime.InteropServices;

namespace Vision.MultiStream.Inference.Services.Cuda
{
    /// <summary>
    /// Phase 4 CUDA 네이티브 DLL(vision_cuda.dll) P/Invoke 브릿지.
    /// DLL 은 Native/Vision.MultiStream.Cuda 프로젝트가 nvcc(CUDA 12.9, sm_61)로 빌드하고,
    /// csproj 의 CopyCudaInferDll 타깃이 출력 폴더로 복사한다(Gpu 빌드 전용).
    /// YOLOv8 전처리(letterbox + 정규화 + CHW)를 CUDA 커널로 실행하는 진입점을 노출한다.
    /// </summary>
    public static class CudaInterop
    {
        private const string Dll = "vision_cuda"; // vision_cuda.dll (출력 폴더에 복사됨)

        // extern "C" int vcuda_preprocess(const unsigned char* bgr, int w, int h,
        //     int newW, int newH, int padX, int padY, float scale, float* out, int outLen)
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)]
        private static extern unsafe int vcuda_preprocess(
            byte* bgr, int width, int height,
            int newW, int newH, int padX, int padY, float scale,
            float* outTensor, int outLen);

        // extern "C" int vcuda_postprocess(const float* output, int numChannels, int numAnchors,
        //     float scale, int padX, int padY, int origW, int origH, float confThreshold,
        //     float* outBoxes, int* outClasses, float* outScores, int maxOut, int* outCount)
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)]
        private static extern unsafe int vcuda_postprocess(
            float* output, int numChannels, int numAnchors,
            float scale, int padX, int padY, int origW, int origH, float confThreshold,
            float* outBoxes, int* outClasses, float* outScores, int maxOut, int* outCount);

        /// <summary>
        /// BGR 픽셀을 CUDA 커널로 전처리해 [1,3,640,640] float 텐서(CHW, 0~1)를 outTensor 에 채운다.
        /// letterbox 파라미터(newW/newH/padX/padY/scale)는 호출자가 계산해 넘긴다
        /// (C# YoloPreprocessor 와 반올림 규칙을 공유해 결과를 일치시키기 위함).
        /// </summary>
        /// <returns>네이티브 반환코드(0=성공, 음수=실패).</returns>
        public static unsafe int Preprocess(
            ReadOnlySpan<byte> bgr, int width, int height,
            int newW, int newH, int padX, int padY, float scale,
            Span<float> outTensor)
        {
            fixed (byte* pSrc = bgr)
            fixed (float* pDst = outTensor)
            {
                return vcuda_preprocess(
                    pSrc, width, height, newW, newH, padX, padY, scale,
                    pDst, outTensor.Length);
            }
        }

        /// <summary>
        /// 추론 raw 출력([1,numChannels,numAnchors])을 CUDA 커널로 필터/디코드/역변환해
        /// 후보 배열(박스 xywh + 클래스 + 점수)을 채운다. NMS 는 호출자(C#)가 수행한다.
        /// </summary>
        /// <returns>네이티브 반환코드(0=성공). outCount 에 임계값 초과 총 후보 수(maxOut 초과 가능).</returns>
        public static unsafe int Postprocess(
            ReadOnlySpan<float> output, int numChannels, int numAnchors,
            float scale, int padX, int padY, int origW, int origH, float confThreshold,
            Span<float> outBoxes, Span<int> outClasses, Span<float> outScores, int maxOut,
            out int outCount)
        {
            fixed (float* pOut = output)
            fixed (float* pBoxes = outBoxes)
            fixed (int* pClasses = outClasses)
            fixed (float* pScores = outScores)
            {
                int count;
                int rc = vcuda_postprocess(
                    pOut, numChannels, numAnchors,
                    scale, padX, padY, origW, origH, confThreshold,
                    pBoxes, pClasses, pScores, maxOut, &count);
                outCount = count;
                return rc;
            }
        }
    }
}
