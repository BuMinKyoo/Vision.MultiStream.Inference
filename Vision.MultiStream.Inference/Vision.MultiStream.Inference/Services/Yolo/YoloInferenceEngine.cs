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
    public enum InferenceDevice { Cpu, DirectML, Gpu }

    public sealed class YoloInferenceEngine : IDisposable
    {
        private const float ConfidenceThreshold = 0.25f;
        private const float IouThreshold = 0.45f;

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
                List<Detection> candidates = ParseOutput(_outputBuffer.AsSpan(0, _outputLen), _numChannels, _numAnchors, lb);
                IReadOnlyList<Detection> detections = ApplyNms(candidates, IouThreshold);
                swPostprocess.Stop();

                return (detections, swInference.Elapsed.TotalMilliseconds, swPostprocess.Elapsed.TotalMilliseconds);
            }
        }

        private static List<Detection> ParseOutput(ReadOnlySpan<float> output, int numChannels, int numAnchors, LetterboxResult lb)
        {
            // output 레이아웃 [1, numChannels, numAnchors] flat (channel-major):
            //   84 = 4(bbox) + 80(COCO 클래스 수), 8400 = anchor 후보 수
            //   원소 [0, c, i] = output[c * numAnchors + i]
            // Tensor 인덱서 대신 raw Span 직접 인덱싱 → 스트라이드 계산/경계검사 제거(70만 회 접근 단축).
            int numClasses = numChannels - 4;

            var candidates = new List<Detection>();

            for (int i = 0; i < numAnchors; i++)
            {
                // 80개 클래스 중 가장 높은 확률의 클래스를 선택
                int bestClass = -1;
                float bestScore = ConfidenceThreshold; // 이 값 미만이면 bestClass가 -1로 남아 건너뜀

                for (int c = 0; c < numClasses; c++)
                {
                    float score = output[(4 + c) * numAnchors + i]; // 앞 4개(bbox)는 건너뛰고 클래스 확률
                    if (score > bestScore)
                    {
                        bestScore = score;
                        bestClass = c;
                    }
                }

                if (bestClass < 0)
                {
                    continue; // 모든 클래스가 임계값 미만 → 버림
                }

                // 모델 출력 좌표는 640×640 letterbox 기준 중심점+크기 형식 (cx, cy, w, h)
                float cx = output[i];                  // 0 * numAnchors + i
                float cy = output[numAnchors + i];     // 1 * numAnchors + i
                float w = output[2 * numAnchors + i];
                float h = output[3 * numAnchors + i];

                // letterbox 좌표 → 원본 이미지 좌표 역변환
                // 1) 중심점에서 좌상단으로 변환: cx - w/2
                // 2) 패딩 제거: - padX (letterbox 회색 여백)
                // 3) 스케일 역산: / scale (리사이즈 되돌리기)
                float x = (cx - w / 2f - lb.PadX) / lb.Scale;
                float y = (cy - h / 2f - lb.PadY) / lb.Scale;
                float bw = w / lb.Scale;
                float bh = h / lb.Scale;

                // 이미지 경계 밖으로 나간 박스를 이미지 안으로 잘라냄
                x = Math.Clamp(x, 0f, lb.OriginalWidth);
                y = Math.Clamp(y, 0f, lb.OriginalHeight);
                bw = Math.Min(bw, lb.OriginalWidth - x);
                bh = Math.Min(bh, lb.OriginalHeight - y);

                if (bw <= 0 || bh <= 0)
                {
                    continue;
                }

                candidates.Add(new Detection
                {
                    X = x,
                    Y = y,
                    Width = bw,
                    Height = bh,
                    ClassId = bestClass,
                    ClassName = CocoLabels.Get(bestClass),
                    Confidence = bestScore
                });
            }

            return candidates;
        }

        private static List<Detection> ApplyNms(List<Detection> input, float iouThreshold)
        {
            // 신뢰도 높은 순으로 정렬 → 높은 것을 기준으로 겹치는 것들을 제거
            var sorted = input.OrderByDescending(d => d.Confidence).ToList();
            var result = new List<Detection>();
            var suppressed = new bool[sorted.Count]; // true면 이미 제거된 박스

            for (int i = 0; i < sorted.Count; i++)
            {
                if (suppressed[i])
                {
                    continue;
                }

                result.Add(sorted[i]); // 살아남은 박스 채택

                // 채택된 박스와 같은 클래스이면서 IoU가 임계값 초과하는 박스는 중복으로 제거
                for (int j = i + 1; j < sorted.Count; j++)
                {
                    if (suppressed[j])
                    {
                        continue;
                    }
                    if (sorted[i].ClassId != sorted[j].ClassId)
                    {
                        continue; // 다른 클래스끼리는 비교 안 함
                    }
                    if (Iou(sorted[i], sorted[j]) > iouThreshold)
                    {
                        suppressed[j] = true; // 겹침이 많으면 낮은 점수 박스 제거
                    }
                }
            }

            return result;
        }

        private static float Iou(Detection a, Detection b)
        {
            // 두 박스의 교집합 영역 좌표 계산
            float x1 = Math.Max(a.X, b.X);
            float y1 = Math.Max(a.Y, b.Y);
            float x2 = Math.Min(a.X + a.Width, b.X + b.Width);
            float y2 = Math.Min(a.Y + a.Height, b.Y + b.Height);

            if (x2 <= x1 || y2 <= y1)
            {
                return 0f; // 겹치는 부분 없음
            }

            float intersection = (x2 - x1) * (y2 - y1);
            float union = a.Width * a.Height + b.Width * b.Height - intersection;

            // IoU = 교집합 / 합집합 (0~1, 1에 가까울수록 두 박스가 거의 같은 위치)
            return intersection / union;
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
