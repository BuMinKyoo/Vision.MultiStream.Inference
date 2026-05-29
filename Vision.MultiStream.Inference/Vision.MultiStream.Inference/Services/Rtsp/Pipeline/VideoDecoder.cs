using System;
using System.Collections.Concurrent;
using System.Threading;
using FFmpeg.AutoGen;
using Vision.MultiStream.Inference.Common;

namespace Vision.MultiStream.Inference.Services.Rtsp.Pipeline
{
    /// <summary>
    /// 파이프라인 2단계(비디오): H.264 등 압축 패킷 → YUV AVFrame 디코딩만 담당.
    /// PTS 게이팅·색변환·표시는 하지 않는다(VideoRenderer 책임).
    /// 디코딩된 프레임은 av_frame_clone(refcount 증가) 으로 넘기고 소비자가 해제한다.
    /// </summary>
    internal sealed unsafe class VideoDecoder : IDisposable
    {
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

        /// <summary>코덱 컨텍스트를 연다. 실패 시 false.</summary>
        public bool Open()
        {
            // Stage 1: HW 디코딩은 UI/배선만 들어왔고 실제 D3D11VA 경로는 Stage 2 에서 구현.
            // 폴백 없이 명시적 실패 처리한다(사용자가 라디오를 SW 로 바꿔야 함).
            if (_useHardwareDecoding)
            {
                _onStatus("하드웨어 디코딩 미구현 (Stage 2 필요) — 라디오를 '소프트웨어' 로 변경하세요");
                return false;
            }

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

            if (ffmpeg.avcodec_open2(_codecCtx, codec, null) < 0)
            {
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

                            // refcount 만 올리는 얕은 복사 → 표시 스레드로 핸드오프.
                            AVFrame* clone = ffmpeg.av_frame_clone(_frame);
                            ffmpeg.av_frame_unref(_frame);

                            if (clone == null)
                            {
                                continue;
                            }

                            try
                            {
                                // 진단: avg 가 높으면 다운스트림(렌더러/UI)이 못 따라와 디코더가 막힘.
                                using (PerfProbe.Measure("rtsp.video.enqueue"))
                                {
                                    _frameQueue.Add((IntPtr)clone, ct);
                                }
                            }
                            catch (InvalidOperationException)
                            {
                                ffmpeg.av_frame_free(&clone);
                            }
                            catch (OperationCanceledException)
                            {
                                ffmpeg.av_frame_free(&clone);
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
                AVCodecContext* c = _codecCtx;
                ffmpeg.avcodec_free_context(&c);
                _codecCtx = null;
            }
        }
    }
}
