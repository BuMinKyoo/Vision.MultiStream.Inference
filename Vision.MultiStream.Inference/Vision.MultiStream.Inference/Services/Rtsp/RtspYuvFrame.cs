using System;
using System.Buffers;

namespace Vision.MultiStream.Inference.Services.Rtsp
{
    /// <summary>
    /// YUV420 표시 프레임. Y/U/V 평면은 ArrayPool 에서 빌린 버퍼다.
    /// LOH 할당/GC 스파이크를 피하려고 매 프레임 new byte[] 대신 풀을 쓰며,
    /// 소비가 끝나면 반드시 Dispose 로 풀에 반납해야 한다.
    ///
    /// 주의: 풀에서 빌린 배열은 요청 크기보다 클 수 있으므로 배열 길이(Length)에
    /// 의존하지 말고 Width/Height/Stride 로 유효 영역을 계산할 것.
    /// </summary>
    public sealed class RtspYuvFrame : IDisposable
    {
        private byte[]? _yPlane;
        private byte[]? _uPlane;
        private byte[]? _vPlane;
        private bool _disposed;

        public RtspYuvFrame(
            byte[] yPlane,
            byte[] uPlane,
            byte[] vPlane,
            int width,
            int height,
            int yStride,
            int uStride,
            int vStride,
            DateTime capturedAt,
            double ptsSeconds)
        {
            _yPlane = yPlane;
            _uPlane = uPlane;
            _vPlane = vPlane;
            Width = width;
            Height = height;
            YStride = yStride;
            UStride = uStride;
            VStride = vStride;
            CapturedAt = capturedAt;
            PtsSeconds = ptsSeconds;
        }

        public byte[] YPlane => _yPlane ?? throw new ObjectDisposedException(nameof(RtspYuvFrame));

        public byte[] UPlane => _uPlane ?? throw new ObjectDisposedException(nameof(RtspYuvFrame));

        public byte[] VPlane => _vPlane ?? throw new ObjectDisposedException(nameof(RtspYuvFrame));

        public int Width { get; }

        public int Height { get; }

        public int YStride { get; }

        public int UStride { get; }

        public int VStride { get; }

        public DateTime CapturedAt { get; }

        public double PtsSeconds { get; }

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }
            _disposed = true;

            // 빌린 배열을 풀에 반납. 콘텐츠는 다음 사용 시 전부 덮어쓰므로 clear 불필요.
            if (_yPlane != null)
            {
                ArrayPool<byte>.Shared.Return(_yPlane);
                _yPlane = null;
            }
            if (_uPlane != null)
            {
                ArrayPool<byte>.Shared.Return(_uPlane);
                _uPlane = null;
            }
            if (_vPlane != null)
            {
                ArrayPool<byte>.Shared.Return(_vPlane);
                _vPlane = null;
            }
        }
    }
}
