using System;
using System.Collections.Concurrent;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Channels;
using FFmpeg.AutoGen;
using Vision.MultiStream.Inference.Services.Audio;
using Vision.MultiStream.Inference.Services.Rtsp.FFmpeg;

namespace Vision.MultiStream.Inference.Services.Rtsp
{
    /// <summary>
    /// RTSP URL 에서 영상/오디오를 받아 처리하는 서비스.
    ///
    /// FFmpeg 정석 3-쓰레드 모델:
    ///   Thread 1 (Demuxer)         : av_read_frame() 으로 RTSP 패킷 수신 후 비디오/오디오 큐로 분기.
    ///   Thread 2 (Video Decoder)   : 비디오 큐 → H.264 디코딩 → sws_scale → BGR → RtspFrame.
    ///   Thread 3 (Audio Decoder)   : 오디오 큐 → AAC/G.711 디코딩 → swr_resample → PCM s16 → IAudioOutput.
    ///
    /// 오디오 출력 정책:
    ///   - 비디오는 항상 활성. 오디오는 비디오에 종속 (audio-only 모드 없음).
    ///   - Start(audioOutput=null) 면 오디오 디코더/오디오 큐 자체를 만들지 않음 → 디먹서가 오디오 패킷 즉시 drop.
    ///   - Start(audioOutput!=null) 면 오디오 스레드 가동 + MediaClock 을 audio-master 모드로 전환.
    ///
    /// A/V 동기화:
    ///   - 오디오 활성: audio-master. 오디오는 게이팅 없이 흘려보내고, 비디오가 실제 오디오 재생 위치에 맞춰 출력.
    ///   - 오디오 비활성: wall-clock. 비디오 첫 프레임 PTS 를 anchor.
    ///
    /// Channel 의 FullMode=DropOldest 로 추론용 큐는 항상 latest-only.
    /// </summary>
    public sealed unsafe class RtspFrameSource : IDisposable
    {
        private readonly string _url;
        private readonly Channel<RtspFrame> _channel;

        private CancellationTokenSource? _cts;
        private Thread? _demuxThread;
        private Thread? _videoThread;
        private Thread? _audioThread;
        private volatile bool _isRunning;

        // 오디오 출력 — 외부에서 주입. null 이면 오디오 비활성.
        private IAudioOutput? _audioOutput;

        // 디먹서가 디코더에게 packet pointer 를 넘기는 큐. 클론한 AVPacket* 를 IntPtr 로 들고 다님.
        private BlockingCollection<IntPtr>? _videoPacketQueue;
        private BlockingCollection<IntPtr>? _audioPacketQueue;

        // 큐 깊이: 너무 깊으면 latency 증가, 너무 얕으면 디먹서 차단됨.
        private const int VideoQueueCapacity = 4;
        private const int AudioQueueCapacity = 32;

        // A/V 동기화 마스터 클럭. 오디오 활성 시 audio-master, 그 외 wall-clock.
        private readonly MediaClock _clock = new();

        private volatile bool _readerFramesEnabled;

        public RtspFrameSource(string rtspUrl)
        {
            _url = rtspUrl;
            _channel = Channel.CreateBounded<RtspFrame>(new BoundedChannelOptions(1)
            {
                FullMode = BoundedChannelFullMode.DropOldest,
                SingleReader = true,
                SingleWriter = true
            });
        }

        public ChannelReader<RtspFrame> Reader => _channel.Reader;
        public bool IsRunning => _isRunning;
        public bool ReaderFramesEnabled
        {
            get => _readerFramesEnabled;
            set => _readerFramesEnabled = value;
        }
        public event EventHandler<string>? StatusChanged;
        public event EventHandler<RtspFrame>? FrameCaptured;

        /// <summary>
        /// 비디오 디코딩 + FrameCaptured/Reader 출력은 항상 활성화된다.
        /// audioOutput 가 null 이 아니면 오디오 디코딩 + 출력도 활성화 (오디오는 비디오에 종속).
        /// </summary>
        public void Start(IAudioOutput? audioOutput)
        {
            if (_isRunning)
            {
                return;
            }

            FFmpegLibraryLoader.EnsureRegistered();

            _audioOutput = audioOutput;
            _clock.Reset(); // 새 세션 시작이므로 클럭 초기화
            _cts = new CancellationTokenSource();

            _videoPacketQueue = new BlockingCollection<IntPtr>(new ConcurrentQueue<IntPtr>(), VideoQueueCapacity);

            _audioPacketQueue = new BlockingCollection<IntPtr>(new ConcurrentQueue<IntPtr>(), AudioQueueCapacity);

            _demuxThread = new Thread(() => DemuxLoop(_cts.Token))
            {
                IsBackground = true,
                Name = $"RtspDemux[{_url}]"
            };
            _isRunning = true;
            _demuxThread.Start();
        }

        public void Stop()
        {
            if (!_isRunning)
            {
                return;
            }

            _isRunning = false;
            _cts?.Cancel();

            // 큐들의 CompleteAdding 으로 디코더 스레드의 GetConsumingEnumerable 이 빠져나오게 한다.
            _videoPacketQueue?.CompleteAdding();
            _audioPacketQueue?.CompleteAdding();

            // 가장 오래 걸릴 수 있는 demux 부터, 그다음 비디오/오디오 디코더.
            _demuxThread?.Join(TimeSpan.FromSeconds(3));
            _videoThread?.Join(TimeSpan.FromSeconds(2));
            _audioThread?.Join(TimeSpan.FromSeconds(2));
            _demuxThread = null;
            _videoThread = null;
            _audioThread = null;

            DrainAndFreePackets(_videoPacketQueue);
            DrainAndFreePackets(_audioPacketQueue);
            _videoPacketQueue?.Dispose();
            _audioPacketQueue?.Dispose();
            _videoPacketQueue = null;
            _audioPacketQueue = null;

            _audioOutput?.Dispose();
            _audioOutput = null;

            _cts?.Dispose();
            _cts = null;
        }

        // ========== Thread 1 : Demuxer ==========

        private void DemuxLoop(CancellationToken ct)
        {
            AVFormatContext* fmtCtx = null;
            AVCodecContext* videoCodecCtx = null;
            AVCodecContext* audioCodecCtx = null;
            AVPacket* packet = null;
            int videoStreamIndex = -1;
            int audioStreamIndex = -1;
            bool needFinalize = true;

            try
            {
                fmtCtx = ffmpeg.avformat_alloc_context();
                if (fmtCtx == null)
                {
                    RaiseStatus("avformat_alloc_context 실패");
                    return;
                }

                // RTSP 옵션: TCP 우선(패킷 손실 회피) + 5초 안에 못 열면 실패
                AVDictionary* opts = null;
                ffmpeg.av_dict_set(&opts, "rtsp_transport", "tcp", 0);
                ffmpeg.av_dict_set(&opts, "stimeout", "5000000", 0); // 마이크로초 단위
                ffmpeg.av_dict_set(&opts, "buffer_size", "1024000", 0);

                // unsafe 컨텍스트에서 필드(fmtCtx)의 주소(&fmtCtx)는 직접 못 잡습니다 (GC가 옮길 수 있는 객체 필드라서) 그래서 복사함
                AVFormatContext* fmtCtxLocal = fmtCtx;

                // 여기서 실제 연결 진행
                int ret = ffmpeg.avformat_open_input(&fmtCtxLocal, _url, null, &opts);
                fmtCtx = fmtCtxLocal;
                if (opts != null)
                {
                    ffmpeg.av_dict_free(&opts);
                }
                if (ret < 0)
                {
                    RaiseStatus($"열기 실패: {_url} ({FfErr(ret)})");
                    return;
                }

                // open_input만 하면 SDP에 적힌 정도의 정보만 있고, 코덱 종류·해상도·프로파일 같은 세부 정보는 비어있는 경우가 많음. find_stream_info가 실제 패킷을 몇 개 미리 읽어서 디코더 파라미터를 채워 넣어줍니다.
                ret = ffmpeg.avformat_find_stream_info(fmtCtx, null);
                if (ret < 0)
                {
                    RaiseStatus($"스트림 정보 조회 실패 ({FfErr(ret)})");
                    return;
                }

                // 스트림 인덱스 확정
                for (int i = 0; i < (int)fmtCtx->nb_streams; i++)
                {
                    AVStream* st = fmtCtx->streams[i];
                    if (st->codecpar->codec_type == AVMediaType.AVMEDIA_TYPE_VIDEO && videoStreamIndex < 0)
                    {
                        videoStreamIndex = i;
                    }
                    else if (st->codecpar->codec_type == AVMediaType.AVMEDIA_TYPE_AUDIO && audioStreamIndex < 0)
                    {
                        audioStreamIndex = i;
                    }
                }

                bool videoActive = videoStreamIndex >= 0 && _videoPacketQueue != null;

                if (videoActive)
                {
                    // ---- Video codec open ----
                    AVStream* vStream = fmtCtx->streams[videoStreamIndex];
                    AVCodec* vCodec = ffmpeg.avcodec_find_decoder(vStream->codecpar->codec_id);
                    if (vCodec == null)
                    {
                        // 비디오 디코더 못 찾으면 오디오도 비활성화됨
                        videoActive = false;
                    }
                    else
                    {
                        videoCodecCtx = ffmpeg.avcodec_alloc_context3(vCodec);
                        ffmpeg.avcodec_parameters_to_context(videoCodecCtx, vStream->codecpar);
                        if (ffmpeg.avcodec_open2(videoCodecCtx, vCodec, null) < 0)
                        {
                            ffmpeg.avcodec_free_context(&videoCodecCtx);
                            videoCodecCtx = null;
                            videoActive = false;
                        }
                    }
                }

                // ---- Audio codec open (오디오 활성 + 스트림 존재 시) ----
                // 오디오는 비디오에 종속 — 비디오가 활성이 아니면 오디오도 열지 않는다.
                bool audioActive = videoActive && _audioOutput != null && audioStreamIndex >= 0 && _audioPacketQueue != null;
                if (audioActive)
                {
                    AVStream* aStream = fmtCtx->streams[audioStreamIndex];
                    AVCodec* aCodec = ffmpeg.avcodec_find_decoder(aStream->codecpar->codec_id);
                    if (aCodec != null)
                    {
                        audioCodecCtx = ffmpeg.avcodec_alloc_context3(aCodec);
                        ffmpeg.avcodec_parameters_to_context(audioCodecCtx, aStream->codecpar);
                        if (ffmpeg.avcodec_open2(audioCodecCtx, aCodec, null) < 0)
                        {
                            // 오디오 코덱 실패해도 영상은 그대로 진행
                            ffmpeg.avcodec_free_context(&audioCodecCtx);
                            audioCodecCtx = null;
                            audioActive = false;
                        }
                    }
                    else
                    {
                        audioActive = false;
                    }
                }

                if (!videoActive && !audioActive)
                {
                    RaiseStatus("열 수 있는 스트림 없음");
                    return;
                }

                string sizeText = videoActive ? $"{videoCodecCtx->width}x{videoCodecCtx->height}" : "-";

                RaiseStatus(
                    $"연결됨: {_url} ({sizeText})" +
                    (videoActive ? " +Video" : "") +
                    (audioActive ? " +Audio" : ""));

                // ---- Decoder threads launch ----
                int videoIdx = videoStreamIndex;
                if (videoActive)
                {
                    AVCodecContext* videoCodecCtxLocal = videoCodecCtx;
                    AVRational vTimeBase = fmtCtx->streams[videoStreamIndex]->time_base;
                    _videoThread = new Thread(() => VideoDecodeLoop(videoCodecCtxLocal, vTimeBase, ct))
                    {
                        IsBackground = true,
                        Name = $"RtspVideo[{_url}]"
                    };
                    _videoThread.Start();
                }

                if (audioActive)
                {
                    AVCodecContext* audioCodecCtxLocal = audioCodecCtx;
                    AVRational aTimeBase = fmtCtx->streams[audioStreamIndex]->time_base;
                    // 오디오 활성 → audio-master 모드로 전환. 비디오는 이 클럭을 따라간다.
                    _clock.UseAudioMaster(_audioOutput!);
                    _audioThread = new Thread(() => AudioDecodeLoop(audioCodecCtxLocal, aTimeBase, ct))
                    {
                        IsBackground = true,
                        Name = $"RtspAudio[{_url}]"
                    };
                    _audioThread.Start();
                }

                // ---- Demux loop ----
                packet = ffmpeg.av_packet_alloc();
                while (!ct.IsCancellationRequested)
                {
                    ret = ffmpeg.av_read_frame(fmtCtx, packet);
                    if (ret < 0)
                    {
                        // EOF 또는 끊김
                        RaiseStatus($"수신 종료 ({FfErr(ret)})");
                        break;
                    }

                    if (videoActive && packet->stream_index == videoIdx)
                    {
                        // av_packet_clone 으로 디코더에 넘길 사본 생성. 디코더가 av_packet_free 책임.
                        AVPacket* clone = ffmpeg.av_packet_clone(packet);
                        if (clone != null)
                        {
                            try
                            {
                                _videoPacketQueue!.Add((IntPtr)clone, ct);
                            }
                            catch (InvalidOperationException)
                            {
                                // CompleteAdding 후 들어옴 → 그냥 free
                                ffmpeg.av_packet_free(&clone);
                            }
                            catch (OperationCanceledException)
                            {
                                ffmpeg.av_packet_free(&clone);
                                break;
                            }
                        }
                    }
                    else if (audioActive && packet->stream_index == audioStreamIndex)
                    {
                        AVPacket* clone = ffmpeg.av_packet_clone(packet);
                        if (clone != null)
                        {
                            try
                            {
                                _audioPacketQueue!.Add((IntPtr)clone, ct);
                            }
                            catch (InvalidOperationException)
                            {
                                ffmpeg.av_packet_free(&clone);
                            }
                            catch (OperationCanceledException)
                            {
                                ffmpeg.av_packet_free(&clone);
                                break;
                            }
                        }
                    }
                    // 다른 스트림(자막 등) 또는 audioActive=false 일 때 오디오 패킷은 그냥 drop

                    ffmpeg.av_packet_unref(packet);
                }
            }
            catch (Exception ex)
            {
                RaiseStatus($"디먹서 오류: {ex.Message}");
            }
            finally
            {
                if (needFinalize)
                {
                    if (packet != null)
                    {
                        ffmpeg.av_packet_free(&packet);
                    }

                    // 디코더 큐 닫고 스레드 종료 대기는 Stop() 에서 처리.
                    // 여기서는 디먹서가 스스로 끝나는 경우(EOF 등)에 대비해 큐만 닫아준다.
                    try
                    {
                        _videoPacketQueue?.CompleteAdding();
                    }
                    catch (ObjectDisposedException) { }

                    try
                    {
                        _audioPacketQueue?.CompleteAdding();
                    }
                    catch (ObjectDisposedException) { }

                    if (videoCodecCtx != null)
                    {
                        ffmpeg.avcodec_free_context(&videoCodecCtx);
                    }
                    if (audioCodecCtx != null)
                    {
                        ffmpeg.avcodec_free_context(&audioCodecCtx);
                    }
                    if (fmtCtx != null)
                    {
                        ffmpeg.avformat_close_input(&fmtCtx);
                    }

                    _channel.Writer.TryComplete();
                    RaiseStatus("정지");
                }
            }
        }

        // ========== Thread 2 : Video Decoder ==========

        private void VideoDecodeLoop(AVCodecContext* codecCtx, AVRational timeBase, CancellationToken ct)
        {
            /*
             
             비디오:                   
            큐: H.264 압축 (~50KB)    
              ↓ avcodec_send_packet   
              ↓ avcodec_receive_frame 
            디코더 출력: YUV420P (3MB)
              ↓ sws_scale (YUV→BGR)   
            변환 결과: BGR24 (6MB)    

             */

            BlockingCollection<IntPtr>? queue = _videoPacketQueue;
            if (queue == null)
            {
                return;
            }

            AVFrame* frame = ffmpeg.av_frame_alloc();
            SwsContext* swsCtx = null;
            int dstBufSize = 0;
            int knownW = 0, knownH = 0;

            try
            {
                // ffmpeg.avcodec_send_packet / avcodec_receive_frame 의 일반적인 루프 구조. 패킷 하나 보낼 때마다 프레임을 여러 개 받을 수 있음 (B-프레임 등).
                foreach (IntPtr pktPtr in queue.GetConsumingEnumerable(ct))
                {
                    AVPacket* pkt = (AVPacket*)pktPtr;
                    try
                    {
                        // 패킷을 디코더에 보냄. 내부 버퍼가 꽉 차면 EAGAIN 반환 → 프레임을 받아서 버퍼 비워주고 다시 시도.
                        if (ffmpeg.avcodec_send_packet(codecCtx, pkt) < 0)
                        {
                            continue;
                        }

                        while (true)
                        {
                            // 디코더에서 프레임을 받음. 아직 처리할 프레임이 없으면 EAGAIN, 스트림 끝나면 EOF 반환.
                            int ret = ffmpeg.avcodec_receive_frame(codecCtx, frame); // 비디오의 경우: frame->data[0]=Y평면, data[1]=U평면, data[2]=V평면, frame->width/height/format 등
                            if (ret == ffmpeg.AVERROR(ffmpeg.EAGAIN) || ret == ffmpeg.AVERROR_EOF)
                            {
                                break;
                            }
                            if (ret < 0)
                            {
                                break;
                            }

                            // ===== PTS 게이트 =====
                            // 마스터 클럭에 맞춰 너무 늦으면 drop, 너무 이르면 sleep.
                            // wall-clock 모드(영상만)면 첫 프레임에서 anchor;
                            // audio-master 모드면 Anchor 는 무시되고 오디오 재생 위치를 따라간다.
                            double ptsSeconds = PtsToSeconds(frame->pts, timeBase);
                            if (!double.IsNaN(ptsSeconds))
                            {
                                if (!_clock.IsReady)
                                {
                                    _clock.Anchor(ptsSeconds);
                                }

                                TimeSpan delay = _clock.GetDelay(ptsSeconds);
                                if (delay < TimeSpan.FromMilliseconds(-100))
                                {
                                    // 너무 늦음 → drop (sws_scale 비용도 절약)
                                    ffmpeg.av_frame_unref(frame);
                                    continue;
                                }
                                if (delay > TimeSpan.FromMilliseconds(5))
                                {
                                    // 너무 이른 → 출력 시점까지 대기 (cancel 시 즉시 빠져나감)
                                    if (ct.WaitHandle.WaitOne(delay))
                                    {
                                        ffmpeg.av_frame_unref(frame);
                                        return;
                                    }
                                }
                            }

                            int w = frame->width;
                            int h = frame->height;

                            // 첫 프레임 또는 해상도 변경 시 sws context 재할당
                            if (swsCtx == null || w != knownW || h != knownH)
                            {
                                if (swsCtx != null)
                                {
                                    ffmpeg.sws_freeContext(swsCtx);
                                    swsCtx = null;
                                }
                                // SWS_BILINEAR == 2 (libswscale.h)
                                swsCtx = ffmpeg.sws_getContext(
                                    w, h, (AVPixelFormat)frame->format,
                                    w, h, AVPixelFormat.AV_PIX_FMT_BGR24,
                                    2, null, null, null);

                                //  BGR24 한 프레임에 필요한 바이트 수 계산
                                dstBufSize = ffmpeg.av_image_get_buffer_size(
                                    AVPixelFormat.AV_PIX_FMT_BGR24, w, h, 1);

                                knownW = w;
                                knownH = h;
                            }

                            // "색공간 변환기 + 화면용 포맷 정리기"
                            // 압축 풀린 픽셀의 모양을 바꾸는 작업
                            IntPtr displayBuffer = Marshal.AllocHGlobal(dstBufSize);
                            byte_ptrArray4 dstData = default;
                            int_array4 dstLinesize = default;
                            ffmpeg.av_image_fill_arrays(
                                ref dstData, ref dstLinesize, (byte*)displayBuffer,
                                AVPixelFormat.AV_PIX_FMT_BGR24, w, h, 1);

                            ffmpeg.sws_scale(
                                swsCtx,
                                frame->data, frame->linesize,
                                0, h,
                                dstData, dstLinesize);

                            // GC 스파이크 완화:
                            // 기존 표시 경로는 프레임마다 full-size managed byte[]를 만들고 BGR 픽셀을 복사했다.
                            // 1080p 기준 프레임당 약 6MB라서 LOH 유입이 빠르게 늘고 Gen2 GC가 자주 발생했다.
                            // 이제 UI 표시용 프레임은 unmanaged buffer를 소유하고, WPF는 WritePixels(IntPtr)로
                            // 바로 읽는다. 그래서 표시만 하는 스트림에서는 매 프레임 managed 대형 배열
                            // 할당과 native->managed 복사를 피할 수 있다.
                            var capturedAt = DateTime.UtcNow;
                            var displayFrame = new RtspFrame(displayBuffer, dstBufSize, w, h, capturedAt, ptsSeconds);
                            bool displayFrameHandedOff = false;
                            try
                            {
                                if (FrameCaptured != null)
                                {
                                    FrameCaptured.Invoke(this, displayFrame);
                                    displayFrameHandedOff = true;
                                }

                                // 현재 추론 API는 byte[] 입력을 사용하므로 추론이 켜진 경우에만
                                // managed 버퍼를 만든다. 추론 OFF 스트림은 UI 표시 경로만 타면서
                                // LOH를 자극하는 대형 byte[] 생성을 하지 않는다.
                                if (_readerFramesEnabled)
                                {
                                    byte[] managed = new byte[dstBufSize];
                                    Marshal.Copy(displayBuffer, managed, 0, dstBufSize);
                                    _channel.Writer.TryWrite(new RtspFrame(managed, w, h, capturedAt, ptsSeconds));
                                }
                            }
                            finally
                            {
                                if (!displayFrameHandedOff)
                                {
                                    displayFrame.Dispose();
                                }
                            }
                            ffmpeg.av_frame_unref(frame);
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
                RaiseStatus($"비디오 디코더 오류: {ex.Message}");
            }
            finally
            {
                if (swsCtx != null)
                {
                    ffmpeg.sws_freeContext(swsCtx);
                }
                ffmpeg.av_frame_free(&frame);
            }
        }

        // ========== Thread 3 : Audio Decoder ==========

        private void AudioDecodeLoop(AVCodecContext* codecCtx, AVRational timeBase, CancellationToken ct)
        {
            /*
             
              오디오:                   
                 큐: AAC 압축 (~700B)      
                   ↓ avcodec_send_packet   
                   ↓ avcodec_receive_frame 
                 디코더 출력: FLTP (8KB)   
                   ↓ swr_convert (FLTP→S16)
                 변환 결과: S16 (4KB)      

             */

            BlockingCollection<IntPtr>? queue = _audioPacketQueue;
            IAudioOutput? output = _audioOutput;
            if (queue == null || output == null)
            {
                return;
            }

            AVFrame* frame = ffmpeg.av_frame_alloc();
            SwrContext* swrCtx = null;

            // 출력 포맷 고정: 16-bit PCM, stereo, 입력과 같은 sample rate (변환 비용 최소화)
            const AVSampleFormat outFormat = AVSampleFormat.AV_SAMPLE_FMT_S16;
            const int outChannels = 2;
            int outSampleRate = 0;
            AVChannelLayout outChLayout = default;
            ffmpeg.av_channel_layout_default(&outChLayout, outChannels);

            // 루프 밖에서 1회만 할당. swr_convert 가 인터리브드 출력 포맷이라 평면 1개로 충분.
            byte** outArr = stackalloc byte*[1];

            try
            {
                foreach (IntPtr pktPtr in queue.GetConsumingEnumerable(ct)) // 여기 들어있는 건 아직 AAC 압축 패킷 (~700바이트)
                {
                    AVPacket* pkt = (AVPacket*)pktPtr;
                    try
                    {
                        if (ffmpeg.avcodec_send_packet(codecCtx, pkt) < 0)
                        {
                            continue;
                        }

                        while (true)
                        {
                            // 이 호출이 끝나야 frame에 FLTP raw 샘플이 들어옴
                            int ret = ffmpeg.avcodec_receive_frame(codecCtx, frame);
                            if (ret == ffmpeg.AVERROR(ffmpeg.EAGAIN) || ret == ffmpeg.AVERROR_EOF)
                            {
                                break;
                            }
                            if (ret < 0)
                            {
                                break;
                            }

                            // 첫 프레임 기준으로 swr 초기화
                            if (swrCtx == null)
                            {
                                outSampleRate = frame->sample_rate;
                                AVChannelLayout inChLayout = frame->ch_layout;
                                SwrContext* swrLocal = null;
                                int alloc = ffmpeg.swr_alloc_set_opts2(
                                    &swrLocal,
                                    &outChLayout, outFormat, outSampleRate,
                                    &inChLayout, (AVSampleFormat)frame->format, frame->sample_rate,
                                    0, null);
                                swrCtx = swrLocal;
                                if (alloc < 0 || swrCtx == null)
                                {
                                    break;
                                }
                                if (ffmpeg.swr_init(swrCtx) < 0)
                                {
                                    break;
                                }
                            }


                            // 입력과 출력이 44.1kHz → 출력 48kHz로 다를 경우에 필요한 부분
                            // 변환 후 샘플 수 추정 (delay 포함해서 안전 마진)
                            long delay = ffmpeg.swr_get_delay(swrCtx, frame->sample_rate); // swr 내부에 남아있는 분수 샘플 (rate 같으면 0)
                            int outSamples = (int)ffmpeg.av_rescale_rnd(
                                delay + frame->nb_samples,  // 비디오의 1프레임당 크기를 구한 것처럼 오디오도 1프레임당 크기를 구하려고 하는것
                                outSampleRate, frame->sample_rate, AVRounding.AV_ROUND_UP);

                            int outBytesPerSample = ffmpeg.av_get_bytes_per_sample(outFormat); // 2 (S16)
                            int maxOutBytes = outSamples * outChannels * outBytesPerSample;
                            byte[] outBuf = new byte[maxOutBytes];

                            int converted;
                            fixed (byte* outBufPtr = outBuf)
                            {
                                outArr[0] = outBufPtr;

                                // 리샘플러 에게 변환 결과를 받아옴
                                // 여기서 FLTP → S16 변환 ( Float [-1.0~1.0] → Int16 [-32768~32767] 변환)
                                converted = ffmpeg.swr_convert( 
                                    swrCtx,
                                    outArr, outSamples, // 출력 버퍼, 변환 후 최대 샘플 수
                                    frame->extended_data, frame->nb_samples); // 입력 버퍼, 변환할 샘플 수
                            }

                            if (converted > 0)
                            {
                                int actualBytes = converted * outChannels * outBytesPerSample;
                                byte[] trimmed;
                                if (actualBytes == outBuf.Length)
                                {
                                    trimmed = outBuf;
                                }
                                else
                                {
                                    trimmed = new byte[actualBytes];
                                    Buffer.BlockCopy(outBuf, 0, trimmed, 0, actualBytes);
                                }

                                double ptsSeconds = PtsToSeconds(frame->pts, timeBase);

                                // Audio-master: 게이팅 없이 바로 push. 사운드카드가 페이싱하고,
                                // 비디오는 이 push PTS 와 출력 버퍼 깊이를 보고 자기를 맞춘다.
                                output.Push(new AudioFrame(trimmed, outSampleRate, outChannels, ptsSeconds));
                                if (!double.IsNaN(ptsSeconds))
                                {
                                    _clock.OnAudioPushed(ptsSeconds);
                                }
                            }

                            ffmpeg.av_frame_unref(frame);
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
                RaiseStatus($"오디오 디코더 오류: {ex.Message}");
            }
            finally
            {
                if (swrCtx != null)
                {
                    SwrContext* tmp = swrCtx;
                    ffmpeg.swr_free(&tmp);
                }
                ffmpeg.av_channel_layout_uninit(&outChLayout);
                ffmpeg.av_frame_free(&frame);
            }
        }

        // ========== helpers ==========

        // FFmpeg pts(time_base 단위) → 초 단위 double 로 환산. NOPTS 면 NaN.
        private static double PtsToSeconds(long pts, AVRational timeBase)
        {
            if (pts == ffmpeg.AV_NOPTS_VALUE)
            {
                return double.NaN;
            }
            return pts * timeBase.num / (double)timeBase.den;
        }

        private static void DrainAndFreePackets(BlockingCollection<IntPtr>? queue)
        {
            if (queue == null)
            {
                return;
            }
            while (queue.TryTake(out IntPtr ptr))
            {
                AVPacket* pkt = (AVPacket*)ptr;
                ffmpeg.av_packet_free(&pkt);
            }
        }

        private static string FfErr(int err)
        {
            const int bufSize = 256;
            byte* buf = stackalloc byte[bufSize];
            ffmpeg.av_strerror(err, buf, (ulong)bufSize);
            return Marshal.PtrToStringAnsi((IntPtr)buf) ?? err.ToString();
        }

        private void RaiseStatus(string message)
        {
            StatusChanged?.Invoke(this, message);
        }

        public void Dispose()
        {
            Stop();
        }
    }
}
