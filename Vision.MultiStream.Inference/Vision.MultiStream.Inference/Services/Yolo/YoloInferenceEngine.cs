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
        }

        /// <summary>
        /// 동기 추론. 호출자 쪽에서 Task.Run 등으로 스레드 분리할 것.
        /// 같은 세션을 여러 스레드에서 동시에 호출하면 안 됨 (직렬화 필요).
        /// </summary>
        public (IReadOnlyList<Detection> Detections, double InferenceMs, double PostprocessMs) Detect(LetterboxResult lb, CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var input = NamedOnnxValue.CreateFromTensor(_inputName, lb.Tensor);

            var swInference = Stopwatch.StartNew();
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = _session.Run(new[] { input });
            swInference.Stop();

            // 출력 텐서: [1, 84, 8400] → (cx, cy, w, h, class0~class79 확률) × 8400개 후보
            Tensor<float> output = results.First().AsTensor<float>();

            var swPostprocess = Stopwatch.StartNew();
            List<Detection> candidates = ParseOutput(output, lb);
            cancellationToken.ThrowIfCancellationRequested();
            IReadOnlyList<Detection> detections = ApplyNms(candidates, IouThreshold);
            swPostprocess.Stop();

            return (detections, swInference.Elapsed.TotalMilliseconds, swPostprocess.Elapsed.TotalMilliseconds);
        }

        private static List<Detection> ParseOutput(Tensor<float> output, LetterboxResult lb)
        {
            // output dims: [1, 84, 8400]
            // 84 = 4(bbox) + 80(COCO 클래스 수)
            // 8400 = 640×640 이미지에서 나오는 anchor 후보 수
            int numChannels = output.Dimensions[1];
            int numAnchors = output.Dimensions[2];
            int numClasses = numChannels - 4;

            var candidates = new List<Detection>();

            for (int i = 0; i < numAnchors; i++)
            {
                // 80개 클래스 중 가장 높은 확률의 클래스를 선택
                int bestClass = -1;
                float bestScore = ConfidenceThreshold; // 이 값 미만이면 bestClass가 -1로 남아 건너뜀

                for (int c = 0; c < numClasses; c++)
                {
                    float score = output[0, 4 + c, i]; // 앞 4개(bbox)는 건너뛰고 클래스 확률
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
                float cx = output[0, 0, i];
                float cy = output[0, 1, i];
                float w = output[0, 2, i];
                float h = output[0, 3, i];

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
            _session?.Dispose();
        }
    }
}
