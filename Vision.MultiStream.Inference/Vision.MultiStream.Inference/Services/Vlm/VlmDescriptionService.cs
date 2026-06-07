using System;
using System.Buffers;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;
using OpenCvSharp;

namespace Vision.MultiStream.Inference.Services.Vlm
{
    /// <summary>
    /// [Step 7] 스트림 1개당 1개. YOLO 가 사람을 검출한 프레임만 VLM 에 넘겨 장면을 묘사하는 파이프라인.
    ///
    /// 흐름: TryTrigger(쿨다운 게이트) → 용량 1 큐(drop) → 백그라운드 워커(BGR→JPEG 인코딩 → IVlmClient).
    /// 추론 루프는 TryTrigger 로 job 하나만 던지고 즉시 반환한다(논블로킹) — VLM 이 느려도 추론/표시는
    /// 막히지 않으며, 처리 중이면 새 요청은 조용히 drop 된다. 실제 VLM 호출은 <see cref="IVlmClient"/> 에 위임.
    /// </summary>
    public sealed class VlmDescriptionService : IDisposable
    {
        private const int JpegQuality = 80;
        private const int MaxEdge = 640; // VLM 입력 긴 변 상한. 비전 토큰 수↓ → 추론 지연·VRAM↓.

        private readonly IVlmClient _client;
        private readonly TimeSpan _cooldown;
        private readonly string _prompt;
        private readonly Channel<VlmJob> _queue;
        private readonly CancellationTokenSource _cts = new();
        private readonly Task _worker;
        private DateTime _lastTriggeredUtc = DateTime.MinValue;

        /// <summary>VLM 요청을 큐에 넣음(트리거 통과 직후, 추론 루프 스레드에서 발생). 진단용.</summary>
        public event Action? RequestQueued;

        /// <summary>VLM 묘사 텍스트 도착(워커 스레드에서 발생 — 구독자가 UI 스레드로 마샬링할 것).</summary>
        public event Action<string>? DescriptionReady;

        /// <summary>VLM 호출 실패(워커 스레드에서 발생).</summary>
        public event Action<string>? ErrorOccurred;

        public VlmDescriptionService(IVlmClient client, TimeSpan cooldown, string prompt)
        {
            _client = client;
            _cooldown = cooldown;
            _prompt = prompt;
            _queue = Channel.CreateBounded<VlmJob>(new BoundedChannelOptions(1)
            {
                FullMode = BoundedChannelFullMode.DropWrite, // 처리 중이면 최신 1장만 남기고 drop
                SingleReader = true,
                SingleWriter = true
            });
            _worker = Task.Run(WorkerLoopAsync);
        }

        /// <summary>
        /// 추론 루프 스레드에서 호출. 논블로킹. "사람이 있고 + 쿨다운이 지났을 때"만 통과시킨다.
        /// bgrPixels 는 호출 직후 호출자가 재사용/반납할 수 있으므로 여기서 즉시 스냅샷을 복사한다.
        /// </summary>
        public void TryTrigger(byte[] bgrPixels, int width, int height, int personCount)
        {
            if (personCount <= 0)
            {
                return;
            }
            DateTime now = DateTime.UtcNow;
            if (now - _lastTriggeredUtc < _cooldown)
            {
                return;
            }
            _lastTriggeredUtc = now;

            int needed = width * height * 3;
            byte[] copy = ArrayPool<byte>.Shared.Rent(needed);
            Array.Copy(bgrPixels, copy, needed);

            if (_queue.Writer.TryWrite(new VlmJob(copy, width, height)))
            {
                RequestQueued?.Invoke();
            }
            else
            {
                ArrayPool<byte>.Shared.Return(copy); // 큐가 꽉 참 → drop, 빌린 버퍼 즉시 반납
            }
        }

        private async Task WorkerLoopAsync()
        {
            try
            {
                await foreach (VlmJob job in _queue.Reader.ReadAllAsync(_cts.Token).ConfigureAwait(false))
                {
                    try
                    {
                        byte[] jpeg = EncodeJpeg(job.BgrBuffer, job.Width, job.Height);
                        string description = await _client.DescribeAsync(jpeg, _prompt, _cts.Token).ConfigureAwait(false);
                        if (!string.IsNullOrWhiteSpace(description))
                        {
                            DescriptionReady?.Invoke(description);
                        }
                    }
                    catch (OperationCanceledException)
                    {
                        throw;
                    }
                    catch (Exception ex)
                    {
                        ErrorOccurred?.Invoke(ex.Message);
                    }
                    finally
                    {
                        ArrayPool<byte>.Shared.Return(job.BgrBuffer);
                    }
                }
            }
            catch (OperationCanceledException)
            {
            }
        }

        // BGR raw → JPEG. 프레임이 이미 OpenCV 네이티브 포맷(BGR)이라 채널 스왑 없이 바로 인코딩.
        // Mat 은 rows*cols*3 바이트만 참조하므로 풀 버퍼가 더 커도 안전하다.
        // 긴 변이 MaxEdge 보다 크면 종횡비 유지 다운스케일(비전 토큰 수↓ → VLM 지연↓).
        private static byte[] EncodeJpeg(byte[] bgrPixels, int width, int height)
        {
            using Mat full = Mat.FromPixelData(height, width, MatType.CV_8UC3, bgrPixels);

            Mat toEncode = full;
            Mat? resized = null;
            int longEdge = Math.Max(width, height);
            if (longEdge > MaxEdge)
            {
                double s = (double)MaxEdge / longEdge;
                resized = new Mat();
                Cv2.Resize(full, resized, new Size((int)Math.Round(width * s), (int)Math.Round(height * s)));
                toEncode = resized;
            }

            Cv2.ImEncode(".jpg", toEncode, out byte[] jpeg, new ImageEncodingParam(ImwriteFlags.JpegQuality, JpegQuality));
            resized?.Dispose();
            return jpeg;
        }

        public void Dispose()
        {
            _cts.Cancel();
            _queue.Writer.TryComplete();
            try
            {
                _worker.Wait(TimeSpan.FromSeconds(2));
            }
            catch
            {
            }
            _cts.Dispose();
        }

        // 큐에 실어 워커로 넘기는 작업 단위. BgrBuffer 는 ArrayPool 에서 빌린 것(워커가 반납).
        private sealed class VlmJob
        {
            public VlmJob(byte[] bgrBuffer, int width, int height)
            {
                BgrBuffer = bgrBuffer;
                Width = width;
                Height = height;
            }

            public byte[] BgrBuffer { get; }
            public int Width { get; }
            public int Height { get; }
        }
    }
}
