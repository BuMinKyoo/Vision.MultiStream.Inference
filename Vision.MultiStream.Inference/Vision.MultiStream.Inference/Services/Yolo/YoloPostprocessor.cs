using System;
using System.Buffers;
using System.Collections.Generic;
using System.Linq;
using Vision.MultiStream.Inference.Models;
using Vision.MultiStream.Inference.Services;
using Vision.MultiStream.Inference.Services.Cuda;

namespace Vision.MultiStream.Inference.Services.Yolo
{
    /// <summary>
    /// YOLOv8 raw 출력 텐서([1, numChannels, numAnchors] = (cx,cy,w,h,class0..) × anchors)를
    /// 검출 리스트로 변환하는 후처리. 책임 1개. 관리 코드(C#) 추론 엔진과
    /// 네이티브(GPU(C++)) 엔진이 동일하게 재사용한다(Step 6-C 에서 분리한 raw Span 인덱싱 그대로).
    /// </summary>
    public static class YoloPostprocessor
    {
        public const float ConfidenceThreshold = 0.25f;
        public const float IouThreshold = 0.45f;

        /// <summary>
        /// raw 출력 → 신뢰도 필터 + 좌표 역변환(ParseOutput) → NMS 까지 한 번에.
        /// </summary>
        public static IReadOnlyList<Detection> Parse(ReadOnlySpan<float> output, int numChannels, int numAnchors, LetterboxResult lb)
        {
            List<Detection> candidates = ParseOutput(output, numChannels, numAnchors, lb);
            return ApplyNms(candidates, IouThreshold);
        }

        // CUDA 후처리에서 GPU 로 넘길 최대 후보 수. YOLOv8 는 임계값 통과 후보가 보통 수십~수백 개라
        // 넉넉히 잡는다(초과분은 커널이 버리고 카운터로 알림 → 실질 손실 거의 없음).
        private const int MaxCudaCandidates = 4096;

        /// <summary>
        /// [Phase 4 Step 14] ParseOutput(앵커 필터+디코드+역변환)을 CUDA 커널로 수행한 뒤,
        /// 소수 후보에 대한 NMS 만 기존 C# ApplyNms 로 처리한다. 결과는 Parse 와 동일해야 한다.
        /// CUDA 실패(rc!=0) 시 전체 CPU 경로(Parse)로 폴백한다.
        /// </summary>
        public static IReadOnlyList<Detection> ParseCuda(ReadOnlySpan<float> output, int numChannels, int numAnchors, LetterboxResult lb)
        {
            const int maxOut = MaxCudaCandidates;
            float[] boxes = ArrayPool<float>.Shared.Rent(maxOut * 4);
            int[] classes = ArrayPool<int>.Shared.Rent(maxOut);
            float[] scores = ArrayPool<float>.Shared.Rent(maxOut);
            try
            {
                int rc = CudaInterop.Postprocess(
                    output, numChannels, numAnchors,
                    lb.Scale, lb.PadX, lb.PadY, lb.OriginalWidth, lb.OriginalHeight, ConfidenceThreshold,
                    boxes.AsSpan(0, maxOut * 4), classes.AsSpan(0, maxOut), scores.AsSpan(0, maxOut), maxOut,
                    out int count);

                if (rc != 0)
                {
                    System.Diagnostics.Debug.WriteLine($"[YoloPostprocessor] CUDA 후처리 실패(rc={rc}) → CPU 폴백");
                    return Parse(output, numChannels, numAnchors, lb);
                }

                int n = Math.Min(count, maxOut);
                var candidates = new List<Detection>(n);
                for (int k = 0; k < n; k++)
                {
                    candidates.Add(new Detection
                    {
                        X = boxes[k * 4 + 0],
                        Y = boxes[k * 4 + 1],
                        Width = boxes[k * 4 + 2],
                        Height = boxes[k * 4 + 3],
                        ClassId = classes[k],
                        ClassName = CocoLabels.Get(classes[k]),
                        Confidence = scores[k]
                    });
                }
                return ApplyNms(candidates, IouThreshold);
            }
            finally
            {
                ArrayPool<float>.Shared.Return(boxes);
                ArrayPool<int>.Shared.Return(classes);
                ArrayPool<float>.Shared.Return(scores);
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
    }
}
