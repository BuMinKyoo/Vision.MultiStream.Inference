using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Vision.MultiStream.Inference.Models;

namespace Vision.MultiStream.Inference.Services.Rtsp
{
    /// <summary>
    /// 동일 ONNX 세션의 동시 추론을 막기 위해 SemaphoreSlim(1,1)로 직렬화하는 데코레이터.
    /// 멀티스트림에서 같은 디바이스를 공유할 때 사용.
    /// </summary>
    public sealed class SerializedFrameDetector : IRtspFrameDetector
    {
        private readonly IRtspFrameDetector _inner;
        private readonly SemaphoreSlim _gate = new(1, 1);

        public SerializedFrameDetector(IRtspFrameDetector inner)
        {
            _inner = inner;
        }

        public async Task<(IReadOnlyList<Detection> Detections, InferenceTimings Timings)> DetectAsync(
            byte[] bgrPixels, int width, int height, CancellationToken cancellationToken = default)
        {
            await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
            try
            {
                return await _inner.DetectAsync(bgrPixels, width, height, cancellationToken).ConfigureAwait(false);
            }
            finally
            {
                _gate.Release();
            }
        }
    }
}
