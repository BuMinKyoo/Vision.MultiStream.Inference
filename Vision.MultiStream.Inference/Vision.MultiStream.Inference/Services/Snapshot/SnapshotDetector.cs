using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Vision.MultiStream.Inference.Models;
using Vision.MultiStream.Inference.Services.Yolo;

namespace Vision.MultiStream.Inference.Services.Snapshot
{
    /// <summary>
    /// 정적 이미지(스냅샷) 도메인 어댑터.
    /// 외부에서 주입된 YoloInferenceEngine 을 공유하며, 자기 자신은 세션을 소유하지 않음
    /// (RTSP 도메인과 같은 엔진을 공유할 수 있도록).
    /// </summary>
    public sealed class SnapshotDetector : ISnapshotDetector
    {
        private readonly YoloInferenceEngine _engine;

        public SnapshotDetector(YoloInferenceEngine engine)
        {
            _engine = engine;
        }

        public Task<IReadOnlyList<Detection>> DetectAsync(string imagePath, CancellationToken cancellationToken = default)
        {
            // Task.Run: UI 스레드를 막지 않도록 전처리 + 추론을 ThreadPool로 분리
            return Task.Run<IReadOnlyList<Detection>>(() =>
            {
                cancellationToken.ThrowIfCancellationRequested();

                // 전처리: 파일 읽기 → letterbox 리사이즈 → 정규화 → CHW 텐서 [1,3,640,640]
                LetterboxResult lb = YoloPreprocessor.Preprocess(imagePath);

                var (detections, _, _) = _engine.Detect(lb, cancellationToken);
                return detections;
            }, cancellationToken);
        }
    }
}
