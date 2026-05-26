using System;
using FFmpeg.AutoGen;

namespace Vision.MultiStream.Inference.Services.Rtsp
{
    /// <summary>
    /// RTSP 수신 스레드가 디코딩해 전달하는 단일 프레임.
    /// 표시 latest-slot 최적화를 위해 managed/unmanaged BGR 뿐 아니라 원본 pixel format도 보존한다.
    /// </summary>
    public sealed class RtspFrame : IDisposable
    {
        private byte[]? _bgrPixels;
        private IntPtr _pixelBuffer;
        private readonly int[]? _planeOffsets;
        private readonly int[]? _lineSizes;
        private bool _disposed;

        public RtspFrame(byte[] bgrPixels, int width, int height, DateTime capturedAt, double ptsSeconds)
        {
            _bgrPixels = bgrPixels;
            Width = width;
            Height = height;
            CapturedAt = capturedAt;
            PtsSeconds = ptsSeconds;
            PixelFormat = AVPixelFormat.AV_PIX_FMT_BGR24;
        }

        public RtspFrame(IntPtr bgrBuffer, int bufferSize, int width, int height, DateTime capturedAt, double ptsSeconds)
            : this(
                bgrBuffer,
                bufferSize,
                width,
                height,
                capturedAt,
                ptsSeconds,
                AVPixelFormat.AV_PIX_FMT_BGR24,
                new[] { width * 3, 0, 0, 0 },
                new[] { 0, 0, 0, 0 })
        {
        }

        public RtspFrame(
            IntPtr pixelBuffer,
            int bufferSize,
            int width,
            int height,
            DateTime capturedAt,
            double ptsSeconds,
            AVPixelFormat pixelFormat,
            int[] lineSizes,
            int[] planeOffsets)
        {
            if (lineSizes.Length != 4)
            {
                throw new ArgumentException("lineSizes must contain exactly 4 items.", nameof(lineSizes));
            }

            if (planeOffsets.Length != 4)
            {
                throw new ArgumentException("planeOffsets must contain exactly 4 items.", nameof(planeOffsets));
            }

            _pixelBuffer = pixelBuffer;
            BufferSize = bufferSize;
            Width = width;
            Height = height;
            CapturedAt = capturedAt;
            PtsSeconds = ptsSeconds;
            PixelFormat = pixelFormat;
            _lineSizes = (int[])lineSizes.Clone();
            _planeOffsets = (int[])planeOffsets.Clone();
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
                    "This frame does not expose managed BGR pixels.");
            }
        }

        public IntPtr BgrBuffer
        {
            get
            {
                if (HasUnmanagedBgrBuffer)
                {
                    return _pixelBuffer;
                }

                throw new InvalidOperationException(
                    "This frame does not expose an unmanaged BGR24 buffer.");
            }
        }

        public int BufferSize { get; }

        public bool HasManagedBgrPixels => _bgrPixels != null;

        public bool HasUnmanagedBuffer => HasUnmanagedBgrBuffer;

        public bool HasUnmanagedBgrBuffer => PixelFormat == AVPixelFormat.AV_PIX_FMT_BGR24 && _pixelBuffer != IntPtr.Zero;

        public bool HasNativePixelBuffer => _pixelBuffer != IntPtr.Zero;

        public AVPixelFormat PixelFormat { get; }

        public int Width { get; }

        public int Height { get; }

        public DateTime CapturedAt { get; }

        public double PtsSeconds { get; }

        public unsafe void FillVideoData(ref byte_ptrArray4 data, ref int_array4 lineSizes)
        {
            if (_pixelBuffer == IntPtr.Zero || _planeOffsets == null || _lineSizes == null)
            {
                throw new InvalidOperationException("This frame does not own an unmanaged pixel buffer.");
            }

            byte* basePtr = (byte*)_pixelBuffer;
            for (int i = 0; i < 4; i++)
            {
                uint index = (uint)i;
                data[index] = _lineSizes[i] > 0 ? basePtr + _planeOffsets[i] : null;
                lineSizes[index] = _lineSizes[i];
            }
        }

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;
            if (_pixelBuffer != IntPtr.Zero)
            {
                System.Runtime.InteropServices.Marshal.FreeHGlobal(_pixelBuffer);
                _pixelBuffer = IntPtr.Zero;
            }
        }
    }
}
