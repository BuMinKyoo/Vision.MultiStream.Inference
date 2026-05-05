using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Vision.MultiStream.Inference.Models;

namespace Vision.MultiStream.Inference.Services.Snapshot
{
    /// <summary>
    /// 디스크 이미지 파일(스냅샷)에서 객체를 검출하는 도메인 추상화.
    /// </summary>
    public interface ISnapshotDetector
    {
        /// <summary>
        /// 입력 이미지 경로의 객체를 검출. 결과 좌표는 원본 이미지 픽셀 기준.
        /// </summary>
        Task<IReadOnlyList<Detection>> DetectAsync(string imagePath, CancellationToken cancellationToken = default);
    }
}
