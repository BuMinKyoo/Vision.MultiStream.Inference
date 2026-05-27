using System;
using System.Collections.Concurrent;
using System.Threading;
using FFmpeg.AutoGen;
using Vision.MultiStream.Inference.Services.Audio;

namespace Vision.MultiStream.Inference.Services.Rtsp.Pipeline
{
    /// <summary>
    /// 파이프라인 2단계(오디오): AAC/G.711 등 → swr_convert 로 S16 stereo PCM 변환.
    /// 변환 결과를 managed AudioFrame 으로 만들어 오디오 프레임 큐에 넣는다.
    /// 사운드카드 push·클럭 갱신은 AudioRenderer 책임.
    /// </summary>
    internal sealed unsafe class AudioDecoder : IDisposable
    {
        private const AVSampleFormat OutFormat = AVSampleFormat.AV_SAMPLE_FMT_S16;
        private const int OutChannels = 2;

        private readonly IntPtr _codecParams; // AVCodecParameters*
        private readonly AVRational _timeBase;
        private readonly BlockingCollection<IntPtr> _packetQueue;
        private readonly BlockingCollection<AudioFrame> _frameQueue;
        private readonly Action<string> _onStatus;

        private AVCodecContext* _codecCtx;
        private AVFrame* _frame;
        private SwrContext* _swrCtx;
        private AVChannelLayout _outChLayout;
        private int _outSampleRate;
        private Thread? _thread;

        public AudioDecoder(
            IntPtr codecParams,
            AVRational timeBase,
            BlockingCollection<IntPtr> packetQueue,
            BlockingCollection<AudioFrame> frameQueue,
            Action<string> onStatus)
        {
            _codecParams = codecParams;
            _timeBase = timeBase;
            _packetQueue = packetQueue;
            _frameQueue = frameQueue;
            _onStatus = onStatus;
        }

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

            if (ffmpeg.avcodec_open2(_codecCtx, codec, null) < 0)
            {
                return false;
            }

            _frame = ffmpeg.av_frame_alloc();
            fixed (AVChannelLayout* layout = &_outChLayout)
            {
                ffmpeg.av_channel_layout_default(layout, OutChannels);
            }
            return true;
        }

        public void Start(CancellationToken ct)
        {
            _thread = new Thread(() => DecodeLoop(ct))
            {
                IsBackground = true,
                Name = "RtspAudioDecode"
            };
            _thread.Start();
        }

        private void DecodeLoop(CancellationToken ct)
        {
            byte** outArr = stackalloc byte*[1];

            try
            {
                foreach (IntPtr pktPtr in _packetQueue.GetConsumingEnumerable(ct))
                {
                    AVPacket* pkt = (AVPacket*)pktPtr;
                    try
                    {
                        if (ffmpeg.avcodec_send_packet(_codecCtx, pkt) < 0)
                        {
                            continue;
                        }

                        while (true)
                        {
                            int ret = ffmpeg.avcodec_receive_frame(_codecCtx, _frame);
                            if (ret == ffmpeg.AVERROR(ffmpeg.EAGAIN) || ret == ffmpeg.AVERROR_EOF)
                            {
                                break;
                            }
                            if (ret < 0)
                            {
                                break;
                            }

                            if (!EnsureSwr())
                            {
                                break;
                            }

                            long delay = ffmpeg.swr_get_delay(_swrCtx, _frame->sample_rate);
                            int outSamples = (int)ffmpeg.av_rescale_rnd(
                                delay + _frame->nb_samples,
                                _outSampleRate, _frame->sample_rate, AVRounding.AV_ROUND_UP);

                            int outBytesPerSample = ffmpeg.av_get_bytes_per_sample(OutFormat);
                            int maxOutBytes = outSamples * OutChannels * outBytesPerSample;
                            byte[] outBuf = new byte[maxOutBytes];

                            int converted;
                            fixed (byte* outBufPtr = outBuf)
                            {
                                outArr[0] = outBufPtr;
                                converted = ffmpeg.swr_convert(
                                    _swrCtx,
                                    outArr, outSamples,
                                    _frame->extended_data, _frame->nb_samples);
                            }

                            if (converted > 0)
                            {
                                int actualBytes = converted * OutChannels * outBytesPerSample;
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

                                double ptsSeconds = FfmpegNative.PtsToSeconds(_frame->pts, _timeBase);
                                try
                                {
                                    _frameQueue.Add(new AudioFrame(trimmed, _outSampleRate, OutChannels, ptsSeconds), ct);
                                }
                                catch (InvalidOperationException)
                                {
                                    // CompleteAdding 후 들어옴 (managed 라 별도 해제 불필요).
                                }
                            }

                            ffmpeg.av_frame_unref(_frame);
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
                _onStatus($"오디오 디코더 오류: {ex.Message}");
            }
            finally
            {
                _frameQueue.CompleteAdding();
            }
        }

        // 첫 프레임 포맷 기준으로 swr 을 1회 초기화한다.
        private bool EnsureSwr()
        {
            if (_swrCtx != null)
            {
                return true;
            }

            _outSampleRate = _frame->sample_rate;
            AVChannelLayout inChLayout = _frame->ch_layout;
            SwrContext* swrLocal = null;

            fixed (AVChannelLayout* outLayout = &_outChLayout)
            {
                int alloc = ffmpeg.swr_alloc_set_opts2(
                    &swrLocal,
                    outLayout, OutFormat, _outSampleRate,
                    &inChLayout, (AVSampleFormat)_frame->format, _frame->sample_rate,
                    0, null);
                _swrCtx = swrLocal;
                if (alloc < 0 || _swrCtx == null)
                {
                    return false;
                }
            }

            return ffmpeg.swr_init(_swrCtx) >= 0;
        }

        public void Join(TimeSpan timeout)
        {
            _thread?.Join(timeout);
            _thread = null;
        }

        public void Dispose()
        {
            if (_swrCtx != null)
            {
                SwrContext* tmp = _swrCtx;
                ffmpeg.swr_free(&tmp);
                _swrCtx = null;
            }
            fixed (AVChannelLayout* layout = &_outChLayout)
            {
                ffmpeg.av_channel_layout_uninit(layout);
            }
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
