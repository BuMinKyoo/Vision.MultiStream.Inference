using System;

namespace Vision.MultiStream.Inference.Services.Rtsp
{
    /// <summary>
    /// RTSP 수신 스레드가 디코딩해서 넘기는 한 프레임 단위.
    /// BGR row-major 픽셀을 managed byte[] 또는 unmanaged buffer 형태로 담는다.
    /// </summary>
    public sealed class RtspFrame : IDisposable
    {
        private byte[]? _bgrPixels;
        private IntPtr _bgrBuffer;
        private bool _disposed;

        // managed BGR 프레임. 현재 추론 channel 경로에서 사용한다.
        public RtspFrame(byte[] bgrPixels, int width, int height, DateTime capturedAt, double ptsSeconds)
        {
            _bgrPixels = bgrPixels;
            Width = width;
            Height = height;
            CapturedAt = capturedAt;
            PtsSeconds = ptsSeconds;
        }

        // unmanaged BGR 프레임. UI 표시 경로에서 매 프레임 대형 byte[] 할당을 피하기 위해 사용한다.
        public RtspFrame(IntPtr bgrBuffer, int bufferSize, int width, int height, DateTime capturedAt, double ptsSeconds)
        {
            _bgrBuffer = bgrBuffer;
            BufferSize = bufferSize;
            Width = width;
            Height = height;
            CapturedAt = capturedAt;
            PtsSeconds = ptsSeconds;
        }

        public byte[] BgrPixels
        {
            get
            {
                if (_bgrPixels != null)
                {
                    return _bgrPixels;
                }

                throw new InvalidOperationException(
                    "This frame uses an unmanaged BGR buffer. Use BgrBuffer instead of BgrPixels.");
            }
        }

        public IntPtr BgrBuffer => _bgrBuffer;

        public int BufferSize { get; }

        public bool HasUnmanagedBuffer => _bgrBuffer != IntPtr.Zero;

        public int Width { get; }

        public int Height { get; }

        public DateTime CapturedAt { get; }

        // PTS가 NOPTS 또는 미지원이면 double.NaN.
        public double PtsSeconds { get; }

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;
            if (_bgrBuffer != IntPtr.Zero)
            {
                System.Runtime.InteropServices.Marshal.FreeHGlobal(_bgrBuffer);
                _bgrBuffer = IntPtr.Zero;
            }
        }
    }
}
