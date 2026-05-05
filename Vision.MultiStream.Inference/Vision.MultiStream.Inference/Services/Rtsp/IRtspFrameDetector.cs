using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Vision.MultiStream.Inference.Models;

namespace Vision.MultiStream.Inference.Services.Rtsp
{
    /// <summary>
    /// RTSP 프레임(메모리상 BGR 픽셀)에서 객체를 검출하는 도메인 추상화.
    /// </summary>
    public interface IRtspFrameDetector
    {
        /// <summary>
        /// bgrPixels: 길이 = width * height * 3, 채널 순서 B-G-R, row-major.
        /// 결과 좌표는 입력 프레임 픽셀 기준.
        /// </summary>
        Task<(IReadOnlyList<Detection> Detections, InferenceTimings Timings)> DetectAsync(byte[] bgrPixels, int width, int height, CancellationToken cancellationToken = default);
    }
}
