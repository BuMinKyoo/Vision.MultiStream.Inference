using System;
using System.Collections.Concurrent;
using System.Threading;
using FFmpeg.AutoGen;

namespace Vision.MultiStream.Inference.Services.Rtsp.Pipeline
{
    /// <summary>
    /// 파이프라인 1단계: RTSP 연결 + 디먹싱.
    /// av_read_frame 으로 받은 패킷을 clone 해서 비디오/오디오 패킷 큐로 라우팅한다.
    /// 코덱은 열지 않는다(디코더 책임). 스트림 파라미터만 RtspStreamInfo 로 노출한다.
    /// </summary>
    internal sealed unsafe class RtspDemuxer : IDisposable
    {
        private readonly string _url;
        private readonly Action<string> _onStatus;

        private AVFormatContext* _fmtCtx;
        private AVPacket* _packet;
        private int _videoStreamIndex = -1;
        private int _audioStreamIndex = -1;

        private Thread? _thread;

        private BlockingCollection<IntPtr>? _videoPacketQueue;
        private BlockingCollection<IntPtr>? _audioPacketQueue;
        private bool _routeAudio;

        public RtspDemuxer(string url, Action<string> onStatus)
        {
            _url = url;
            _onStatus = onStatus;
        }

        /// <summary>
        /// 연결 + 스트림 탐색을 동기로 수행한다. 실패하면 HasVideo=false 인 정보를 돌려준다.
        /// 성공 시 AVFormatContext 는 열린 채 유지되어 read 루프에서 사용된다.
        /// </summary>
        public RtspStreamInfo Open()
        {
            var info = new RtspStreamInfo();

            _fmtCtx = ffmpeg.avformat_alloc_context();
            if (_fmtCtx == null)
            {
                _onStatus("avformat_alloc_context 실패");
                return info;
            }

            AVDictionary* opts = null;
            ffmpeg.av_dict_set(&opts, "rtsp_transport", "tcp", 0);
            ffmpeg.av_dict_set(&opts, "stimeout", "5000000", 0);
            ffmpeg.av_dict_set(&opts, "buffer_size", "1024000", 0);

            AVFormatContext* fmtCtxLocal = _fmtCtx;
            int ret = ffmpeg.avformat_open_input(&fmtCtxLocal, _url, null, &opts);
            _fmtCtx = fmtCtxLocal;
            if (opts != null)
            {
                ffmpeg.av_dict_free(&opts);
            }
            if (ret < 0)
            {
                _onStatus($"열기 실패: {_url} ({FfmpegNative.FfErr(ret)})");
                return info;
            }

            ret = ffmpeg.avformat_find_stream_info(_fmtCtx, null);
            if (ret < 0)
            {
                _onStatus($"스트림 정보 조회 실패 ({FfmpegNative.FfErr(ret)})");
                return info;
            }

            for (int i = 0; i < (int)_fmtCtx->nb_streams; i++)
            {
                AVStream* st = _fmtCtx->streams[i];
                if (st->codecpar->codec_type == AVMediaType.AVMEDIA_TYPE_VIDEO && _videoStreamIndex < 0)
                {
                    _videoStreamIndex = i;
                }
                else if (st->codecpar->codec_type == AVMediaType.AVMEDIA_TYPE_AUDIO && _audioStreamIndex < 0)
                {
                    _audioStreamIndex = i;
                }
            }

            if (_videoStreamIndex >= 0)
            {
                AVStream* vStream = _fmtCtx->streams[_videoStreamIndex];
                info.HasVideo = true;
                info.VideoStreamIndex = _videoStreamIndex;
                info.VideoCodecParams = (IntPtr)vStream->codecpar;
                info.VideoTimeBase = vStream->time_base;
                info.Width = vStream->codecpar->width;
                info.Height = vStream->codecpar->height;
            }

            if (_audioStreamIndex >= 0)
            {
                AVStream* aStream = _fmtCtx->streams[_audioStreamIndex];
                info.HasAudio = true;
                info.AudioStreamIndex = _audioStreamIndex;
                info.AudioCodecParams = (IntPtr)aStream->codecpar;
                info.AudioTimeBase = aStream->time_base;
            }

            return info;
        }

        /// <summary>
        /// read 루프 스레드 시작. routeAudio=false 면 오디오 패킷은 즉시 drop 한다.
        /// </summary>
        public void Start(
            BlockingCollection<IntPtr> videoPacketQueue,
            BlockingCollection<IntPtr>? audioPacketQueue,
            bool routeAudio,
            CancellationToken ct)
        {
            _videoPacketQueue = videoPacketQueue;
            _audioPacketQueue = audioPacketQueue;
            _routeAudio = routeAudio && audioPacketQueue != null;

            _thread = new Thread(() => ReadLoop(ct))
            {
                IsBackground = true,
                Name = $"RtspDemux[{_url}]"
            };
            _thread.Start();
        }

        private void ReadLoop(CancellationToken ct)
        {
            try
            {
                _packet = ffmpeg.av_packet_alloc();
                while (!ct.IsCancellationRequested)
                {
                    int ret = ffmpeg.av_read_frame(_fmtCtx, _packet);
                    if (ret < 0)
                    {
                        _onStatus($"수신 종료 ({FfmpegNative.FfErr(ret)})");
                        break;
                    }

                    if (_packet->stream_index == _videoStreamIndex)
                    {
                        Route(_videoPacketQueue, ct);
                    }
                    else if (_routeAudio && _packet->stream_index == _audioStreamIndex)
                    {
                        Route(_audioPacketQueue, ct);
                    }

                    ffmpeg.av_packet_unref(_packet);
                }
            }
            catch (OperationCanceledException)
            {
            }
            catch (Exception ex)
            {
                _onStatus($"디먹서 오류: {ex.Message}");
            }
            finally
            {
                // 더 이상 패킷이 들어오지 않음을 디코더에게 알린다 (EOF·오류 포함).
                _videoPacketQueue?.CompleteAdding();
                _audioPacketQueue?.CompleteAdding();
            }
        }

        // 현재 패킷을 clone 해서 큐에 넣는다. 큐가 받지 않으면(완료/취소) clone 을 해제한다.
        private void Route(BlockingCollection<IntPtr>? queue, CancellationToken ct)
        {
            if (queue == null)
            {
                return;
            }

            AVPacket* clone = ffmpeg.av_packet_clone(_packet);
            if (clone == null)
            {
                return;
            }

            try
            {
                queue.Add((IntPtr)clone, ct);
            }
            catch (InvalidOperationException)
            {
                // CompleteAdding 후 들어옴.
                ffmpeg.av_packet_free(&clone);
            }
            catch (OperationCanceledException)
            {
                ffmpeg.av_packet_free(&clone);
                throw;
            }
        }

        public void Join(TimeSpan timeout)
        {
            _thread?.Join(timeout);
            _thread = null;
        }

        public void Dispose()
        {
            if (_packet != null)
            {
                AVPacket* p = _packet;
                ffmpeg.av_packet_free(&p);
                _packet = null;
            }
            if (_fmtCtx != null)
            {
                AVFormatContext* f = _fmtCtx;
                ffmpeg.avformat_close_input(&f);
                _fmtCtx = null;
            }
        }
    }
}
