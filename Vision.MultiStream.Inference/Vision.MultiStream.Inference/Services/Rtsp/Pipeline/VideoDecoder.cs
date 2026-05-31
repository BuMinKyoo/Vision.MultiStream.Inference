using System;
using System.Collections.Concurrent;
using System.Runtime.InteropServices;
using System.Threading;
using FFmpeg.AutoGen;
using Vision.MultiStream.Inference.Common;

namespace Vision.MultiStream.Inference.Services.Rtsp.Pipeline
{
    /// <summary>
    /// 파이프라인 2단계(비디오): H.264 등 압축 패킷 → YUV AVFrame 디코딩만 담당.
    /// PTS 게이팅·색변환·표시는 하지 않는다(VideoRenderer 책임).
    /// 디코딩된 프레임은 av_frame_clone(refcount 증가) 으로 넘기고 소비자가 해제한다.
    /// SW 경로: codec → YUV420P 프레임을 그대로 큐로 전달.
    /// HW(D3D11VA) 경로: codec → AV_PIX_FMT_D3D11 (GPU) 프레임 → av_hwframe_transfer_data
    ///   로 NV12 sw 프레임 다운로드 → sw 프레임을 큐로 전달. 폴백 없음(실패 시 즉시 에러).
    /// </summary>
    internal sealed unsafe class VideoDecoder : IDisposable
    {
        // get_format 콜백을 FFmpeg 가 호출하려면 함수 포인터가 살아 있어야 한다.
        // delegate 인스턴스를 static readonly 로 보관해 GC 수거를 막는다.
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate AVPixelFormat GetFormatDelegate(AVCodecContext* ctx, AVPixelFormat* fmts);

        private static readonly GetFormatDelegate _getFormatDelegate = HwGetFormatCallback;
        private static readonly IntPtr _getFormatPtr =
            Marshal.GetFunctionPointerForDelegate(_getFormatDelegate);

        // FFmpeg 가 후보 픽셀 포맷 리스트를 주면 D3D11 을 골라 HW 디코딩으로 진입.
        // D3D11 을 고르기 전에 hw_frames_ctx 를 직접 구성한다(아래 TrySetupD3D11Frames).
        // 자동 생성에 맡기면 디코드 텍스처가 BIND_DECODER 만 달려 셰이더로 샘플(SRV)할 수 없어
        // 컴포지터가 표시하지 못한다. 없으면 첫 SW fmt 를 돌려준다.
        private static AVPixelFormat HwGetFormatCallback(AVCodecContext* ctx, AVPixelFormat* fmts)
        {
            for (AVPixelFormat* p = fmts; *p != AVPixelFormat.AV_PIX_FMT_NONE; p++)
            {
                if (*p == AVPixelFormat.AV_PIX_FMT_D3D11)
                {
                    TrySetupD3D11Frames(ctx);
                    return AVPixelFormat.AV_PIX_FMT_D3D11;
                }
            }
            return *fmts;
        }

        // 디코드 풀(hw_frames_ctx) 을 직접 만들어 BIND_SHADER_RESOURCE 를 추가한다.
        // → 컴포지터가 디코드 NV12 텍스처에 SRV 를 만들어 셰이더로 직접 샘플 가능.
        // 또한 in-flight(프레임 큐 + 컴포지터 latest + 디코드 중) 보유분만큼 풀을 키워 고갈을 막는다.
        private static void TrySetupD3D11Frames(AVCodecContext* ctx)
        {
            if (ctx->hw_device_ctx == null)
            {
                return;
            }

            AVBufferRef* frames = null;
            int ret = ffmpeg.avcodec_get_hw_frames_parameters(
                ctx, ctx->hw_device_ctx, AVPixelFormat.AV_PIX_FMT_D3D11, &frames);
            if (ret < 0 || frames == null)
            {
                return;
            }

            AVHWFramesContext* fctx = (AVHWFramesContext*)frames->data;
            // 프레임 큐(10) + 컴포지터 latest(1) + 디코드 중(1) 보유분 헤드룸.
            fctx->initial_pool_size += 16;

            AVD3D11VAFramesContext* d3d11 = (AVD3D11VAFramesContext*)fctx->hwctx;
            d3d11->BindFlags |= HwDeviceContext.FramesShaderResourceBindFlag;

            if (ffmpeg.av_hwframe_ctx_init(frames) < 0)
            {
                ffmpeg.av_buffer_unref(&frames);
                return;
            }
            // 소유권을 codec ctx 로 넘긴다(avcodec_free_context 가 unref).
            ctx->hw_frames_ctx = frames;
        }

        private readonly IntPtr _codecParams; // AVCodecParameters*
        private readonly BlockingCollection<IntPtr> _packetQueue;
        private readonly BlockingCollection<IntPtr> _frameQueue;
        private readonly Action<string> _onStatus;
        private readonly bool _useHardwareDecoding;

        private AVCodecContext* _codecCtx;
        private AVFrame* _frame;
        private Thread? _thread;

        public VideoDecoder(
            IntPtr codecParams,
            BlockingCollection<IntPtr> packetQueue,
            BlockingCollection<IntPtr> frameQueue,
            Action<string> onStatus,
            bool useHardwareDecoding = false)
        {
            _codecParams = codecParams;
            _packetQueue = packetQueue;
            _frameQueue = frameQueue;
            _onStatus = onStatus;
            _useHardwareDecoding = useHardwareDecoding;
        }

        /// <summary>코덱 컨텍스트를 연다. 실패 시 false. HW 경로는 폴백 없이 그대로 실패.</summary>
        public bool Open()
        {
            AVCodecParameters* codecpar = (AVCodecParameters*)_codecParams;
            AVCodec* codec = ffmpeg.avcodec_find_decoder(codecpar->codec_id);
            if (codec == null)
            {
                return false;
            }

            _codecCtx = ffmpeg.avcodec_alloc_context3(codec);
            if (_codecCtx == null)
            {
                return false;
            }

            if (ffmpeg.avcodec_parameters_to_context(_codecCtx, codecpar) < 0)
            {
                return false;
            }

            if (_useHardwareDecoding)
            {
                // HwDeviceContext.AcquireRef() 는 이미 av_buffer_ref 결과를 돌려준다.
                // codec ctx 가 소유권을 가져가 avcodec_free_context 시점에 unref 한다 →
                // 우리는 별도 보관 X (이중 unref 방지).
                IntPtr deviceRef = HwDeviceContext.AcquireRef();
                if (deviceRef == IntPtr.Zero)
                {
                    _onStatus("하드웨어 디코딩 실패: D3D11VA 디바이스 생성 불가");
                    return false;
                }

                _codecCtx->hw_device_ctx = (AVBufferRef*)deviceRef;
                _codecCtx->get_format = new AVCodecContext_get_format_func { Pointer = _getFormatPtr };
            }

            if (ffmpeg.avcodec_open2(_codecCtx, codec, null) < 0)
            {
                if (_useHardwareDecoding)
                {
                    _onStatus("하드웨어 디코딩 실패: avcodec_open2");
                }
                return false;
            }

            _frame = ffmpeg.av_frame_alloc();
            return true;
        }

        public void Start(CancellationToken ct)
        {
            _thread = new Thread(() => DecodeLoop(ct))
            {
                IsBackground = true,
                Name = "RtspVideoDecode"
            };
            _thread.Start();
        }

        private void DecodeLoop(CancellationToken ct)
        {
            try
            {
                foreach (IntPtr pktPtr in _packetQueue.GetConsumingEnumerable(ct))
                {
                    AVPacket* pkt = (AVPacket*)pktPtr;
                    try
                    {
                        int sendRet;
                        // 진단: H.264 디코딩 CPU 비용 (send).
                        using (PerfProbe.Measure("rtsp.video.send"))
                        {
                            sendRet = ffmpeg.avcodec_send_packet(_codecCtx, pkt);
                        }
                        if (sendRet < 0)
                        {
                            continue;
                        }

                        while (true)
                        {
                            int ret;
                            // 진단: H.264 디코딩 CPU 비용 (receive). count = 초당 디코드 프레임 수.
                            using (PerfProbe.Measure("rtsp.video.receive"))
                            {
                                ret = ffmpeg.avcodec_receive_frame(_codecCtx, _frame);
                            }
                            if (ret == ffmpeg.AVERROR(ffmpeg.EAGAIN) || ret == ffmpeg.AVERROR_EOF)
                            {
                                break;
                            }
                            if (ret < 0)
                            {
                                break;
                            }

                            // SW(YUV420P) 든 HW(D3D11 GPU 텍스처) 든 refcount 만 올리는 얕은 복사로
                            // 표시 스레드에 핸드오프한다. HW 는 CPU 다운로드 없이 GPU 프레임을 그대로 넘긴다
                            // (다운로드/색변환은 표시 경로에서 추론이 필요할 때만 일어난다).
                            AVFrame* handoff = ffmpeg.av_frame_clone(_frame);
                            ffmpeg.av_frame_unref(_frame);
                            if (handoff == null)
                            {
                                continue;
                            }

                            try
                            {
                                // 진단: avg 가 높으면 다운스트림(렌더러/UI)이 못 따라와 디코더가 막힘.
                                using (PerfProbe.Measure("rtsp.video.enqueue"))
                                {
                                    _frameQueue.Add((IntPtr)handoff, ct);
                                }
                            }
                            catch (InvalidOperationException)
                            {
                                ffmpeg.av_frame_free(&handoff);
                            }
                            catch (OperationCanceledException)
                            {
                                ffmpeg.av_frame_free(&handoff);
                                throw;
                            }
                        }
                    }
                    finally
                    {
                        ffmpeg.av_packet_free(&pkt);
                    }
                }
            }
            catch (OperationCanceledException)
            {
            }
            catch (Exception ex)
            {
                _onStatus($"비디오 디코더 오류: {ex.Message}");
            }
            finally
            {
                _frameQueue.CompleteAdding();
            }
        }

        public void Join(TimeSpan timeout)
        {
            _thread?.Join(timeout);
            _thread = null;
        }

        public void Dispose()
        {
            if (_frame != null)
            {
                AVFrame* f = _frame;
                ffmpeg.av_frame_free(&f);
                _frame = null;
            }
            if (_codecCtx != null)
            {
                // codec ctx 가 hw_device_ctx 의 ref 소유권을 가지고 있어 free_context 가 unref.
                AVCodecContext* c = _codecCtx;
                ffmpeg.avcodec_free_context(&c);
                _codecCtx = null;
            }
        }
    }
}
