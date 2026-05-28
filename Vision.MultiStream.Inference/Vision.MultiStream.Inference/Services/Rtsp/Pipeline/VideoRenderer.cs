using System;
using System.Buffers;
using System.Collections.Concurrent;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Channels;
using FFmpeg.AutoGen;
using Vision.MultiStream.Inference.Common;

namespace Vision.MultiStream.Inference.Services.Rtsp.Pipeline
{
    /// <summary>
    /// 비디오 표시 경로의 런타임 토글. 표시 스레드(읽기)와 UI 스레드(쓰기)가 공유하므로 volatile.
    /// </summary>
    public sealed class VideoRenderSettings
    {
        private volatile bool _readerFramesEnabled;
        private volatile bool _useYuvDisplayFrames;

        public bool ReaderFramesEnabled
        {
            get => _readerFramesEnabled;
            set => _readerFramesEnabled = value;
        }

        public bool UseYuvDisplayFrames
        {
            get => _useYuvDisplayFrames;
            set => _useYuvDisplayFrames = value;
        }
    }

    /// <summary>
    /// 파이프라인 3단계(비디오): 디코딩된 YUV 프레임을 받아 MediaClock 으로 페이싱한 뒤
    /// (1) YUV 표시 프레임 발행, (2) sws_scale 로 BGR24 변환 후 표시/추론 출력을 만든다.
    /// 입력 프레임(AVFrame*)의 소유권을 가지므로 처리 후 av_frame_free 한다.
    /// </summary>
    internal sealed unsafe class VideoRenderer : IDisposable
    {
        private readonly BlockingCollection<IntPtr> _frameQueue;
        private readonly AVRational _timeBase;
        private readonly MediaClock _clock;
        private readonly VideoRenderSettings _settings;
        private readonly ChannelWriter<RtspFrame> _inferenceWriter;
        private readonly Func<RtspFrame, bool> _raiseFrameCaptured;
        private readonly Action<RtspYuvFrame> _raiseYuvCaptured;
        private readonly Action<string> _onStatus;

        private SwsContext* _swsCtx;
        private int _knownW;
        private int _knownH;
        private int _dstBufSize;
        private Thread? _thread;

        public VideoRenderer(
            BlockingCollection<IntPtr> frameQueue,
            AVRational timeBase,
            MediaClock clock,
            VideoRenderSettings settings,
            ChannelWriter<RtspFrame> inferenceWriter,
            Func<RtspFrame, bool> raiseFrameCaptured,
            Action<RtspYuvFrame> raiseYuvCaptured,
            Action<string> onStatus)
        {
            _frameQueue = frameQueue;
            _timeBase = timeBase;
            _clock = clock;
            _settings = settings;
            _inferenceWriter = inferenceWriter;
            _raiseFrameCaptured = raiseFrameCaptured;
            _raiseYuvCaptured = raiseYuvCaptured;
            _onStatus = onStatus;
        }

        public void Start(CancellationToken ct)
        {
            _thread = new Thread(() => RenderLoop(ct))
            {
                IsBackground = true,
                Name = "RtspVideoRender"
            };
            _thread.Start();
        }

        private void RenderLoop(CancellationToken ct)
        {
            try
            {
                foreach (IntPtr framePtr in _frameQueue.GetConsumingEnumerable(ct))
                {
                    AVFrame* frame = (AVFrame*)framePtr;
                    try
                    {
                        if (!Present(frame, ct))
                        {
                            // 취소로 인한 조기 종료.
                            return;
                        }
                    }
                    finally
                    {
                        AVFrame* f = frame;
                        ffmpeg.av_frame_free(&f);
                    }
                }
            }
            catch (OperationCanceledException)
            {
            }
            catch (Exception ex)
            {
                _onStatus($"비디오 렌더러 오류: {ex.Message}");
            }
        }

        // 한 프레임을 페이싱·변환·발행한다. 취소로 빠져나가야 하면 false.
        private bool Present(AVFrame* frame, CancellationToken ct)
        {
            double ptsSeconds = FfmpegNative.PtsToSeconds(frame->pts, _timeBase);

            // ===== PTS 게이트 =====
            if (!double.IsNaN(ptsSeconds))
            {
                if (!_clock.IsReady)
                {
                    _clock.Anchor(ptsSeconds);
                }

                TimeSpan delay = _clock.GetDelay(ptsSeconds);
                if (delay < TimeSpan.FromMilliseconds(-100))
                {
                    // 너무 늦음 → drop (sws_scale 비용 절약).
                    return true;
                }
                if (delay > TimeSpan.FromMilliseconds(5))
                {
                    if (ct.WaitHandle.WaitOne(delay))
                    {
                        return false;
                    }
                }
            }

            int w = frame->width;
            int h = frame->height;
            var capturedAt = DateTime.UtcNow;

            if (_settings.UseYuvDisplayFrames && IsYuv420Frame(frame))
            {
                using (PerfProbe.Measure("rtsp.yuv420.copy"))
                {
                    _raiseYuvCaptured(CopyYuv420Frame(frame, capturedAt, ptsSeconds));
                }

                // 추론이 꺼져 있으면 BGR 변환은 건너뛴다.
                if (!_settings.ReaderFramesEnabled)
                {
                    return true;
                }
            }

            // 첫 프레임 또는 해상도 변경 시 sws context 재할당.
            if (_swsCtx == null || w != _knownW || h != _knownH)
            {
                if (_swsCtx != null)
                {
                    ffmpeg.sws_freeContext(_swsCtx);
                    _swsCtx = null;
                }
                _swsCtx = ffmpeg.sws_getContext(
                    w, h, (AVPixelFormat)frame->format,
                    w, h, AVPixelFormat.AV_PIX_FMT_BGR24,
                    2, null, null, null);
                _dstBufSize = ffmpeg.av_image_get_buffer_size(AVPixelFormat.AV_PIX_FMT_BGR24, w, h, 1);
                _knownW = w;
                _knownH = h;
            }

            IntPtr displayBuffer = Marshal.AllocHGlobal(_dstBufSize);
            byte_ptrArray4 dstData = default;
            int_array4 dstLinesize = default;
            ffmpeg.av_image_fill_arrays(
                ref dstData, ref dstLinesize, (byte*)displayBuffer,
                AVPixelFormat.AV_PIX_FMT_BGR24, w, h, 1);

            using (PerfProbe.Measure("rtsp.sws_scale.bgr24"))
            {
                ffmpeg.sws_scale(_swsCtx, frame->data, frame->linesize, 0, h, dstData, dstLinesize);
            }

            // 추론용 managed 복사는 이벤트 raise 보다 먼저 한다.
            // raise 안에서 핸들러가 동기로 displayBuffer 를 해제할 수 있어 use-after-free 위험이 있음.
            byte[]? managedForInference = null;
            if (_settings.ReaderFramesEnabled)
            {
                managedForInference = new byte[_dstBufSize];
                Marshal.Copy(displayBuffer, managedForInference, 0, _dstBufSize);
            }

            var displayFrame = new RtspFrame(displayBuffer, _dstBufSize, w, h, capturedAt, ptsSeconds);
            bool handedOff = false;
            try
            {
                using (PerfProbe.Measure("rtsp.frame_captured.invoke"))
                {
                    handedOff = _raiseFrameCaptured(displayFrame);
                }

                if (managedForInference != null)
                {
                    _inferenceWriter.TryWrite(new RtspFrame(managedForInference, w, h, capturedAt, ptsSeconds));
                }
            }
            finally
            {
                if (!handedOff)
                {
                    displayFrame.Dispose();
                }
            }

            return true;
        }

        private static bool IsYuv420Frame(AVFrame* frame)
        {
            AVPixelFormat format = (AVPixelFormat)frame->format;
            return format == AVPixelFormat.AV_PIX_FMT_YUV420P
                || format == AVPixelFormat.AV_PIX_FMT_YUVJ420P;
        }

        private static RtspYuvFrame CopyYuv420Frame(AVFrame* frame, DateTime capturedAt, double ptsSeconds)
        {
            int width = frame->width;
            int height = frame->height;
            int chromaWidth = (width + 1) / 2;
            int chromaHeight = (height + 1) / 2;

            // LOH 할당/GC 스파이크 회피: 매 프레임 new byte[] 대신 풀에서 빌린다.
            // RtspYuvFrame.Dispose 에서 풀로 반납된다. 빌린 배열은 요청보다 클 수 있으니
            // 유효 영역은 width*height 로만 다룬다(Stride==width 로 타이트 패킹).
            byte[] yPlane = ArrayPool<byte>.Shared.Rent(width * height);
            byte[] uPlane = ArrayPool<byte>.Shared.Rent(chromaWidth * chromaHeight);
            byte[] vPlane = ArrayPool<byte>.Shared.Rent(chromaWidth * chromaHeight);

            CopyPlane(frame->data[0], frame->linesize[0], width, height, yPlane);
            CopyPlane(frame->data[1], frame->linesize[1], chromaWidth, chromaHeight, uPlane);
            CopyPlane(frame->data[2], frame->linesize[2], chromaWidth, chromaHeight, vPlane);

            return new RtspYuvFrame(
                yPlane, uPlane, vPlane,
                width, height,
                width, chromaWidth, chromaWidth,
                capturedAt, ptsSeconds);
        }

        // source(unmanaged, sourceStride) → dest 앞쪽에 width 단위로 타이트하게 복사.
        // dest 는 풀에서 빌린 (더 클 수 있는) 버퍼.
        private static void CopyPlane(byte* source, int sourceStride, int width, int height, byte[] dest)
        {
            fixed (byte* destinationBase = dest)
            {
                byte* destination = destinationBase;
                for (int y = 0; y < height; y++)
                {
                    Buffer.MemoryCopy(source + (y * sourceStride), destination, width, width);
                    destination += width;
                }
            }
        }

        public void Join(TimeSpan timeout)
        {
            _thread?.Join(timeout);
            _thread = null;
        }

        public void Dispose()
        {
            if (_swsCtx != null)
            {
                ffmpeg.sws_freeContext(_swsCtx);
                _swsCtx = null;
            }
        }
    }
}
