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
        private volatile bool _useCompositorDisplay;

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

        // 표시 경로 선택. true=컴포지터(YuvFrameCaptured), false=개별 per-stream(YuvIndividualFrameCaptured).
        // VideoRenderer.Present 가 이 값으로 어느 이벤트로 YUV 프레임을 발행할지 분기한다.
        public bool UseCompositorDisplay
        {
            get => _useCompositorDisplay;
            set => _useCompositorDisplay = value;
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
        private readonly Action<RtspYuvFrame> _raiseYuvIndividualCaptured;
        private readonly Action<RtspD3D11Frame> _raiseD3D11Captured;
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
            Action<RtspYuvFrame> raiseYuvIndividualCaptured,
            Action<RtspD3D11Frame> raiseD3D11Captured,
            Action<string> onStatus)
        {
            _frameQueue = frameQueue;
            _timeBase = timeBase;
            _clock = clock;
            _settings = settings;
            _inferenceWriter = inferenceWriter;
            _raiseFrameCaptured = raiseFrameCaptured;
            _raiseYuvCaptured = raiseYuvCaptured;
            _raiseYuvIndividualCaptured = raiseYuvIndividualCaptured;
            _raiseD3D11Captured = raiseD3D11Captured;
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
                    AVFrame* originalFrame = (AVFrame*)framePtr;

                    // HW(D3D11) 프레임: CPU 다운로드 없이 GPU 텍스처를 컴포지터로 직접 핸드오프.
                    // 이 분기는 프레임 소유권을 PresentHardware 가 가져간다(여기서 free 하지 않음).
                    if (originalFrame->format == (int)AVPixelFormat.AV_PIX_FMT_D3D11)
                    {
                        if (!PresentHardware(originalFrame, ct))
                        {
                            return;
                        }
                        continue;
                    }

                    // SW 경로: YUV420P 프레임을 페이싱·변환·발행하고 여기서 소유권을 해제한다.
                    try
                    {
                        if (!Present(originalFrame, ct))
                        {
                            // 취소로 인한 조기 종료.
                            return;
                        }
                    }
                    finally
                    {
                        AVFrame* f = originalFrame;
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

        // HW(D3D11) 프레임을 페이싱한 뒤 GPU 텍스처 그대로 컴포지터로 핸드오프한다.
        // 프레임 소유권은 이 메서드가 가져간다(drop/취소 시 free, 표시 시 RtspD3D11Frame 으로 이전).
        // 취소로 빠져나가야 하면 false.
        private bool PresentHardware(AVFrame* frame, CancellationToken ct)
        {
            double ptsSeconds = FfmpegNative.PtsToSeconds(frame->pts, _timeBase);

            // ===== PTS 게이트 (SW 경로와 동일) =====
            if (!double.IsNaN(ptsSeconds))
            {
                if (!_clock.IsReady)
                {
                    _clock.Anchor(ptsSeconds);
                }

                TimeSpan delay = _clock.GetDelay(ptsSeconds);
                if (delay < TimeSpan.FromMilliseconds(-100))
                {
                    // 너무 늦음 → drop.
                    FreeFrame(frame);
                    return true;
                }
                if (delay > TimeSpan.FromMilliseconds(5))
                {
                    if (ct.WaitHandle.WaitOne(delay))
                    {
                        FreeFrame(frame);
                        return false;
                    }
                }
            }

            // 추론 ON 인 스트림만 GPU→CPU 다운로드 + BGR 변환(드문 경로).
            if (_settings.ReaderFramesEnabled)
            {
                EmitInferenceFromHw(frame, ptsSeconds);
            }

            // 표시: GPU 프레임을 그대로 컴포지터로. RtspD3D11Frame 이 frame ref 소유권을 가져간다
            // (해상도(ImageWidth/Height) 알림은 표시 경로 VM 핸들러가 처리).
            var d3d11 = new RtspD3D11Frame((IntPtr)frame, ptsSeconds);
            _raiseD3D11Captured(d3d11);
            return true;
        }

        private static void FreeFrame(AVFrame* frame)
        {
            AVFrame* f = frame;
            ffmpeg.av_frame_free(&f);
        }

        // 추론용: HW 프레임을 NV12 sw frame 으로 다운로드한 뒤 BGR24 로 변환해 추론 채널로 보낸다.
        private void EmitInferenceFromHw(AVFrame* hw, double ptsSeconds)
        {
            AVFrame* sw = ffmpeg.av_frame_alloc();
            if (sw == null)
            {
                return;
            }
            try
            {
                if (ffmpeg.av_hwframe_transfer_data(sw, hw, 0) < 0)
                {
                    return;
                }
                ffmpeg.av_frame_copy_props(sw, hw);
                EmitBgrForInference(sw, ptsSeconds);
            }
            finally
            {
                ffmpeg.av_frame_free(&sw);
            }
        }

        // sw frame(보통 NV12) → BGR24 변환 후 추론 채널로 전달. 표시는 컴포지터가 담당하므로 여기선 추론만.
        private void EmitBgrForInference(AVFrame* frame, double ptsSeconds)
        {
            int w = frame->width;
            int h = frame->height;
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
            if (_swsCtx == null)
            {
                return;
            }

            IntPtr buffer = Marshal.AllocHGlobal(_dstBufSize);
            try
            {
                byte_ptrArray4 dstData = default;
                int_array4 dstLinesize = default;
                ffmpeg.av_image_fill_arrays(
                    ref dstData, ref dstLinesize, (byte*)buffer,
                    AVPixelFormat.AV_PIX_FMT_BGR24, w, h, 1);

                using (PerfProbe.Measure("rtsp.sws_scale.bgr24"))
                {
                    ffmpeg.sws_scale(_swsCtx, frame->data, frame->linesize, 0, h, dstData, dstLinesize);
                }

                byte[] managed = new byte[_dstBufSize];
                Marshal.Copy(buffer, managed, 0, _dstBufSize);
                _inferenceWriter.TryWrite(new RtspFrame(managed, w, h, DateTime.UtcNow, ptsSeconds));
            }
            finally
            {
                Marshal.FreeHGlobal(buffer);
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
                    RtspYuvFrame yuv = CopyYuv420Frame(frame, capturedAt, ptsSeconds);

                    // 표시 경로 분기: 컴포지터 모드면 컴포지터 이벤트로, 개별 모드면 별도 개별 이벤트로 발행한다.
                    // 구독자(StreamItemViewModel)는 각 이벤트에 전용 핸들러 하나씩만 붙으므로 핸들러 내부 분기가 없다.
                    if (_settings.UseCompositorDisplay)
                    {
                        _raiseYuvCaptured(yuv);
                    }
                    else
                    {
                        _raiseYuvIndividualCaptured(yuv);
                    }
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
