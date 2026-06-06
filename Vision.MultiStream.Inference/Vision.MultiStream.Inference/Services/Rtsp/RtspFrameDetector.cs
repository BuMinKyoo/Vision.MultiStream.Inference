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

                try
                {
                    var (detections, inferenceMs, postprocessMs) = _engine.Detect(lb, cancellationToken);

                    var timings = new InferenceTimings
                    {
                        PreprocessMs = swPreprocess.Elapsed.TotalMilliseconds,
                        InferenceMs = inferenceMs,
                        PostprocessMs = postprocessMs,
                    };

                    return (detections, timings);
                }
                finally
                {
                    // 텐서 백킹(풀 버퍼)을 ArrayPool 로 반납. _engine.Detect 는 동기로
                    // _session.Run 까지 끝낸 뒤이므로 입력 메모리를 더 참조하지 않는다.
                    lb.Dispose();
                }
            }, cancellationToken);
        }
    }
}
