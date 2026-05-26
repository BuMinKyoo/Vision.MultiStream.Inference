using System;
using System.Runtime.InteropServices;
using FFmpeg.AutoGen;

namespace Vision.MultiStream.Inference.Services.Rtsp
{
    /// <summary>
    /// 표시 시점에만 native decoded frame을 재사용 가능한 BGR24 버퍼로 변환한다.
    /// latest-slot에서 실제로 소비되는 프레임에만 sws_scale 비용을 쓰기 위한 용도다.
    /// </summary>
    public sealed unsafe class RtspDisplayFrameConverter : IDisposable
    {
        private SwsContext* _swsCtx;
        private AVPixelFormat _sourcePixelFormat = AVPixelFormat.AV_PIX_FMT_NONE;
        private int _sourceWidth;
        private int _sourceHeight;
        private IntPtr _bgrBuffer;
        private int _bgrBufferSize;
        private byte_ptrArray4 _dstData;
        private int_array4 _dstLineSizes;

        public IntPtr ConvertToBgr24(RtspFrame frame, out int bufferSize, out int stride)
        {
            if (!frame.HasNativePixelBuffer)
            {
                throw new InvalidOperationException("A native pixel buffer frame is required for conversion.");
            }

            EnsureConversionState(frame);

            byte_ptrArray4 srcData = default;
            int_array4 srcLineSizes = default;
            frame.FillVideoData(ref srcData, ref srcLineSizes);

            ffmpeg.sws_scale(
                _swsCtx,
                srcData,
                srcLineSizes,
                0,
                frame.Height,
                _dstData,
                _dstLineSizes);

            bufferSize = _bgrBufferSize;
            stride = _dstLineSizes[(uint)0];
            return _bgrBuffer;
        }

        public void Reset()
        {
            if (_swsCtx != null)
            {
                ffmpeg.sws_freeContext(_swsCtx);
                _swsCtx = null;
            }

            if (_bgrBuffer != IntPtr.Zero)
            {
                Marshal.FreeHGlobal(_bgrBuffer);
                _bgrBuffer = IntPtr.Zero;
            }

            _bgrBufferSize = 0;
            _sourceWidth = 0;
            _sourceHeight = 0;
            _sourcePixelFormat = AVPixelFormat.AV_PIX_FMT_NONE;
            _dstData = default;
            _dstLineSizes = default;
        }

        public void Dispose()
        {
            Reset();
        }

        private void EnsureConversionState(RtspFrame frame)
        {
            bool reinitialize =
                _swsCtx == null
                || _sourceWidth != frame.Width
                || _sourceHeight != frame.Height
                || _sourcePixelFormat != frame.PixelFormat;

            if (reinitialize)
            {
                Reset();

                _swsCtx = ffmpeg.sws_getContext(
                    frame.Width,
                    frame.Height,
                    frame.PixelFormat,
                    frame.Width,
                    frame.Height,
                    AVPixelFormat.AV_PIX_FMT_BGR24,
                    2,
                    null,
                    null,
                    null);

                if (_swsCtx == null)
                {
                    throw new InvalidOperationException(
                        $"sws_getContext failed for {frame.PixelFormat} -> BGR24.");
                }

                _sourceWidth = frame.Width;
                _sourceHeight = frame.Height;
                _sourcePixelFormat = frame.PixelFormat;
            }

            int requiredSize = ffmpeg.av_image_get_buffer_size(
                AVPixelFormat.AV_PIX_FMT_BGR24,
                frame.Width,
                frame.Height,
                1);

            if (requiredSize <= 0)
            {
                throw new InvalidOperationException("Failed to compute BGR24 buffer size.");
            }

            if (_bgrBuffer == IntPtr.Zero || _bgrBufferSize != requiredSize)
            {
                if (_bgrBuffer != IntPtr.Zero)
                {
                    Marshal.FreeHGlobal(_bgrBuffer);
                }

                _bgrBuffer = Marshal.AllocHGlobal(requiredSize);
                _bgrBufferSize = requiredSize;

                ffmpeg.av_image_fill_arrays(
                    ref _dstData,
                    ref _dstLineSizes,
                    (byte*)_bgrBuffer,
                    AVPixelFormat.AV_PIX_FMT_BGR24,
                    frame.Width,
                    frame.Height,
                    1);
            }
        }
    }
}
