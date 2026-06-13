using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Threading;
using Vision.MultiStream.Inference.Models;

namespace Vision.MultiStream.Inference.Services.Yolo
{
    /// <summary>
    /// Phase 3 "GPU(C++)" 추론 엔진. 네이티브 DLL(vision_infer.dll)의 ONNX Runtime + DirectML
    /// 세션으로 추론한다. 전처리(<see cref="YoloPreprocessor"/>)와 후처리(<see cref="YoloPostprocessor"/>)는
    /// 관리 코드 경로와 동일하게 재사용하고, 가운데 ORT 실행만 네이티브로 넘긴다("추론 엔진만").
    ///
    /// 경계: C# 가 만든 입력 텐서([1,3,640,640] float)와 재사용 출력 버퍼를 unsafe 포인터로 핀해
    /// 네이티브에 그대로 넘긴다(프레임당 신규 할당 0, C# IoBinding 출력 재사용과 동일 철학).
    /// </summary>
    public sealed class NativeYoloEngine : IYoloEngine, IDisposable
    {
        // vision_infer.dll 은 csproj 빌드 타깃이 C# 출력 폴더로 복사한다(onnxruntime.dll 과 동일 폴더).
        private const string Dll = "vision_infer";

        [DllImport(Dll, CharSet = CharSet.Unicode)]
        private static extern IntPtr vinfer_create([MarshalAs(UnmanagedType.LPWStr)] string modelPath, int device);

        [DllImport(Dll)]
        private static extern unsafe int vinfer_infer(IntPtr handle, float* input, float* output, int outputLen);

        [DllImport(Dll)]
        private static extern int vinfer_output_dims(IntPtr handle, out int numChannels, out int numAnchors);

        [DllImport(Dll)]
        private static extern void vinfer_destroy(IntPtr handle);

        // 네이티브 device 코드: 1 = DirectML.
        private const int DeviceDirectML = 1;

        private readonly IntPtr _handle;
        private readonly int _numChannels;   // 출력 dim[1] (= 4 + 클래스 수)
        private readonly int _numAnchors;    // 출력 dim[2] (= 8400)
        private readonly int _outputLen;     // numChannels * numAnchors
        private readonly float[] _outputBuffer; // 1회 할당, 매 프레임 재사용 (raw 출력 백킹)

        // 출력 버퍼/네이티브 세션은 공유 가변 상태 → 추론~파싱을 락으로 직렬화(상위 SerializedFrameDetector
        // 와 별개로 방어적). 네이티브 내부도 mutex 로 보호하지만 출력 버퍼는 여기서 보호해야 안전.
        private readonly object _runLock = new object();

        public InferenceDevice Device => InferenceDevice.NativeCpp;

        public NativeYoloEngine(string modelPath)
        {
            _handle = vinfer_create(modelPath, DeviceDirectML);
            if (_handle == IntPtr.Zero)
            {
                throw new InvalidOperationException(
                    "네이티브 추론 엔진(vision_infer.dll) 초기화 실패: 모델 로드 또는 DirectML 초기화 오류.");
            }

            int rc = vinfer_output_dims(_handle, out _numChannels, out _numAnchors);
            if (rc != 0 || _numChannels <= 0 || _numAnchors <= 0)
            {
                vinfer_destroy(_handle);
                throw new InvalidOperationException($"네이티브 출력 형상 조회 실패 (rc={rc}).");
            }

            _outputLen = _numChannels * _numAnchors;
            _outputBuffer = new float[_outputLen];
        }

        public (IReadOnlyList<Detection> Detections, double InferenceMs, double PostprocessMs) Detect(
            LetterboxResult lb, CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();

            lock (_runLock)
            {
                // 입력: C# 전처리가 만든 [1,3,640,640] 텐서 백킹(풀 버퍼). 출력: 재사용 버퍼.
                // 둘 다 호출 동안 핀해 네이티브가 in-place 로 읽고/쓴다.
                ReadOnlySpan<float> input = lb.Tensor.Buffer.Span;

                var swInference = Stopwatch.StartNew();
                int rc;
                unsafe
                {
                    fixed (float* pIn = input)
                    fixed (float* pOut = _outputBuffer)
                    {
                        // 추론
                        rc = vinfer_infer(_handle, pIn, pOut, _outputLen);
                    }
                }
                swInference.Stop();

                if (rc != 0)
                {
                    throw new InvalidOperationException($"네이티브 추론 실패 (rc={rc}).");
                }

                cancellationToken.ThrowIfCancellationRequested();

                var swPostprocess = Stopwatch.StartNew();

                // 후처리
                IReadOnlyList<Detection> detections = YoloPostprocessor.Parse(
                    _outputBuffer.AsSpan(0, _outputLen), _numChannels, _numAnchors, lb);
                swPostprocess.Stop();

                return (detections, swInference.Elapsed.TotalMilliseconds, swPostprocess.Elapsed.TotalMilliseconds);
            }
        }

        public void Dispose()
        {
            if (_handle != IntPtr.Zero)
            {
                vinfer_destroy(_handle);
            }
        }
    }
}
