using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Vision.MultiStream.Inference.Models;
using Vision.MultiStream.Inference.Services;

namespace Vision.MultiStream.Inference.Services.Yolo
{
    /// <summary>
    /// YOLOv8 ONNX 모델의 raw 추론 엔진. 책임 1개:
    /// "이미 만들어진 LetterboxResult 를 받아 검출 결과 리스트를 돌려준다".
    /// 도메인(정적 이미지/RTSP) 무관. InferenceSession 의 단일 소유자.
    /// 출력 텐서: [1, 84, 8400] = (cx, cy, w, h, class0..class79) × 8400 후보.
    /// </summary>
    // NativeCpp: Phase 3 "GPU(C++)" — 네이티브 DLL(vision_infer.dll) + DirectML 로 추론.
    public enum InferenceDevice { Cpu, DirectML, Gpu, NativeCpp }

    public sealed class YoloInferenceEngine : IYoloEngine, IDisposable
    {
        private readonly InferenceSession _session;
        private readonly string _inputName;

        // ===== IoBinding (출력 버퍼 재사용) =====
        // 레거시 Run(inputs) 는 매 추론마다 출력 텐서([1,84,8400] ≈ 2.7MB)를 새로 할당해 LOH/Gen2 GC
        // 스파이크를 유발한다. IoBinding 으로 출력 OrtValue 를 고정 버퍼에 한 번만 바인드해 재사용하면
        // 추론 출력의 프레임당 관리 힙 할당이 사라진다.
        private readonly string _outputName;
        private readonly int _numChannels;       // 출력 dim[1] (= 4 + 클래스 수)
        private readonly int _numAnchors;        // 출력 dim[2] (= 8400)
        private readonly int _outputLen;         // dim 곱 = 재사용 버퍼 길이
        private readonly long[] _inputShape;     // [1,3,640,640]
        private readonly long[] _outputShape;    // [1,84,8400]
        private readonly float[] _outputBuffer;  // 1회 할당, 매 프레임 재사용 (출력 백킹)
        private readonly OrtValue _outputOrt;    // _outputBuffer 위의 텐서 값 (고정 바인드)
        private readonly OrtIoBinding _binding;
        private readonly RunOptions _runOptions;

        // 출력 버퍼/바인딩은 공유 가변 상태. 엔진이 여러 스트림에 공유되거나 (_dmlEngine ?? _cpuEngine
        // 처럼) aliasing 되어 서로 다른 직렬화 게이트가 같은 엔진을 동시에 호출해도 안전하도록 락으로 보호.
        private readonly object _runLock = new object();

        public InferenceDevice Device { get; }

        public YoloInferenceEngine(string modelPath, InferenceDevice device = InferenceDevice.Cpu)
        {
            Device = device;
            var options = new SessionOptions();

            if (device == InferenceDevice.DirectML)
            {
                try
                {
                    // DirectML: Windows 내장 DirectX 12 ML API 사용 (CUDA Toolkit 불필요)
                    // DirectX 12 지원 GPU면 NVIDIA/AMD/Intel 모두 동작
                    options.AppendExecutionProvider_DML(0);
                }
                catch (Exception ex)
                {
                    throw new InvalidOperationException($"DirectML 초기화 실패: {ex.Message}", ex);
                }
            }
            else if (device == InferenceDevice.Gpu)
            {
                try
                {
                    // CUDA: CUDA Toolkit 12.x 설치 필요
                    options.AppendExecutionProvider_CUDA(0);
                }
                catch (Exception ex)
                {
                    throw new InvalidOperationException($"GPU(CUDA) 초기화 실패: {ex.Message}", ex);
                }
            }

            _session = new InferenceSession(modelPath, options);
            _inputName = _session.InputMetadata.Keys.First();

            // 출력 형상을 메타데이터에서 읽어 재사용 버퍼/OrtValue 를 1회 준비.
            // 고정 입력(640)인 YOLOv8 export 는 출력도 고정([1,84,8400]).
            _outputName = _session.OutputMetadata.Keys.First();
            int[] dims = _session.OutputMetadata[_outputName].Dimensions;
            if (dims.Length != 3 || dims[0] <= 0 || dims[1] <= 0 || dims[2] <= 0)
            {
                throw new InvalidOperationException(
                    "IoBinding 출력 사전 바인딩에는 고정 출력 형상이 필요합니다. " +
                    $"현재 출력 '{_outputName}' 형상: [{string.Join(",", dims)}]. " +
                    "동적 형상 모델이면 입력 크기를 고정해 export 하세요.");
            }

            _numChannels = dims[1];
            _numAnchors = dims[2];
            _outputLen = dims[0] * dims[1] * dims[2];
            _inputShape = new long[] { 1, 3, YoloPreprocessor.InputSize, YoloPreprocessor.InputSize };
            _outputShape = new long[] { dims[0], dims[1], dims[2] };

            _outputBuffer = new float[_outputLen];
            _outputOrt = OrtValue.CreateTensorValueFromMemory(_outputBuffer, _outputShape);

            _binding = _session.CreateIoBinding();
            _binding.BindOutput(_outputName, _outputOrt); // 출력은 고정 버퍼 → 1회 바인드로 충분
            _runOptions = new RunOptions();
        }

        /// <summary>
        /// 동기 추론. 호출자 쪽에서 Task.Run 등으로 스레드 분리할 것.
        /// 같은 세션을 여러 스레드에서 동시에 호출하면 안 됨 (직렬화 필요).
        /// </summary>
        public (IReadOnlyList<Detection> Detections, double InferenceMs, double PostprocessMs) Detect(LetterboxResult lb, CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();

            // 출력 버퍼/바인딩이 공유 상태이므로 추론~파싱(출력 버퍼 읽기)을 락으로 보호.
            // RunWithBinding 이 _outputBuffer 에 결과를 in-place 로 채우고, ParseOutput 이 그 즉시
            // 후보로 복사해 빠져나가므로 다음 호출이 버퍼를 덮어써도 안전하다.
            lock (_runLock)
            {
                // 입력: 풀에서 빌린 텐서 메모리를 그대로 가리키는 OrtValue (관리 힙 신규 할당 없음).
                using OrtValue inputOrt = OrtValue.CreateTensorValueFromMemory(
                    OrtMemoryInfo.DefaultInstance, lb.Tensor.Buffer, _inputShape);
                _binding.BindInput(_inputName, inputOrt);

                var swInference = Stopwatch.StartNew();
                _session.RunWithBinding(_runOptions, _binding); // 출력은 _outputBuffer 에 재기록
                swInference.Stop();

                cancellationToken.ThrowIfCancellationRequested();

                // 출력 텐서: [1, 84, 8400] → (cx, cy, w, h, class0~class79 확률) × 8400개 후보
                var swPostprocess = Stopwatch.StartNew();
                IReadOnlyList<Detection> detections = YoloPostprocessor.Parse(
                    _outputBuffer.AsSpan(0, _outputLen), _numChannels, _numAnchors, lb);
                swPostprocess.Stop();

                return (detections, swInference.Elapsed.TotalMilliseconds, swPostprocess.Elapsed.TotalMilliseconds);
            }
        }

        public void Dispose()
        {
            _binding?.Dispose();
            _outputOrt?.Dispose();
            _runOptions?.Dispose();
            _session?.Dispose();
        }
    }
}
