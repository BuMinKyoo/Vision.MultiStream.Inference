using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using Vision.MultiStream.Inference.Common;
using Vision.MultiStream.Inference.Models;
using Vision.MultiStream.Inference.Services.Yolo;

namespace Vision.MultiStream.Inference.Services.Rtsp
{
    /// <summary>
    /// RTSP 프레임(메모리상 BGR 픽셀)에서 객체를 검출하는 도메인 어댑터.
    /// 외부에서 주입된 추론 엔진(<see cref="IYoloEngine"/>: 관리 코드 ONNX 또는 네이티브 GPU(C++))을 공유.
    /// </summary>
    public sealed class RtspFrameDetector : IRtspFrameDetector
    {
        private readonly IYoloEngine _engine;
        // true 면 전처리를 C# CPU(YoloPreprocessor.Preprocess) 대신 CUDA 커널로 수행(Phase 4 Step 14).
        private readonly bool _useCudaPreprocess;

        public RtspFrameDetector(IYoloEngine engine, bool useCudaPreprocess = false)
        {
            _engine = engine;
            _useCudaPreprocess = useCudaPreprocess;
        }

        public Task<(IReadOnlyList<Detection> Detections, InferenceTimings Timings)> DetectAsync(byte[] bgrPixels, int width, int height, CancellationToken cancellationToken = default)
        {
            return Task.Run(() =>
            {
                cancellationToken.ThrowIfCancellationRequested();

                var swPreprocess = Stopwatch.StartNew();
                LetterboxResult lb = _useCudaPreprocess
                    ? YoloPreprocessor.PreprocessCuda(bgrPixels, width, height)
                    : YoloPreprocessor.Preprocess(bgrPixels, width, height);
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

                    // 추론 경로 타이밍을 콘솔(PerfProbe) 로그에도 1초 단위 집계로 출력.
                    // 화면(StreamItemViewModel)과 동일 데이터지만, RTSP 디코딩 probe 와 같은 형식으로
                    // 한곳에서 비교하기 위함. 스트림이 여러 개면 같은 이름으로 합산 집계된다.
                    PerfProbe.RecordMs(_useCudaPreprocess ? "rtsp.infer.preprocess.cuda" : "rtsp.infer.preprocess", timings.PreprocessMs);
                    PerfProbe.RecordMs("rtsp.infer.inference", timings.InferenceMs);
                    PerfProbe.RecordMs("rtsp.infer.postprocess", timings.PostprocessMs);

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
