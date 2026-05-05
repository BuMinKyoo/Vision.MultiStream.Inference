using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using Vision.MultiStream.Inference.Models;
using Vision.MultiStream.Inference.Services.Yolo;

namespace Vision.MultiStream.Inference.Services.Rtsp
{
    /// <summary>
    /// RTSP 프레임(메모리상 BGR 픽셀)에서 객체를 검출하는 도메인 어댑터.
    /// 외부에서 주입된 YoloInferenceEngine 을 공유.
    /// </summary>
    public sealed class RtspFrameDetector : IRtspFrameDetector
    {
        private readonly YoloInferenceEngine _engine;

        public RtspFrameDetector(YoloInferenceEngine engine)
        {
            _engine = engine;
        }

        public Task<(IReadOnlyList<Detection> Detections, InferenceTimings Timings)> DetectAsync(byte[] bgrPixels, int width, int height, CancellationToken cancellationToken = default)
        {
            return Task.Run(() =>
            {
                cancellationToken.ThrowIfCancellationRequested();

                var swPreprocess = Stopwatch.StartNew();
                LetterboxResult lb = YoloPreprocessor.Preprocess(bgrPixels, width, height);
                swPreprocess.Stop();

                var (detections, inferenceMs, postprocessMs) = _engine.Detect(lb, cancellationToken);

                var timings = new InferenceTimings
                {
                    PreprocessMs = swPreprocess.Elapsed.TotalMilliseconds,
                    InferenceMs = inferenceMs,
                    PostprocessMs = postprocessMs,
                };

                return (detections, timings);
            }, cancellationToken);
        }
    }
}
