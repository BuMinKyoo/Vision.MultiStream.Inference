using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

namespace Vision.MultiStream.Inference.Services.Vlm
{
    /// <summary>
    /// 로컬 Ollama 서버의 비전 모델(qwen2.5vl 계열)을 HTTP 로 호출하는 <see cref="IVlmClient"/> 구현.
    /// 사전 준비(코드 외): 터미널에서 `ollama pull qwen2.5vl:3b` 후 Ollama 데몬 기동(기본 11434 포트).
    /// 실제 사용 모델명은 호출부(StreamItemViewModel)에서 생성자 인자로 주입한다(현재 qwen2.5vl:3b).
    ///
    /// 요청 (POST /api/generate): { "model":"qwen2.5vl:3b", "prompt":"...", "images":["&lt;base64 JPEG&gt;"], "stream":false }
    /// 응답: { "response":"장면 묘사 텍스트", ... }
    /// </summary>
    public sealed class OllamaVlmClient : IVlmClient
    {
        // VLM 추론은 수백 ms~수 초 걸릴 수 있어 타임아웃을 넉넉히. HttpClient 는 비용이 커 정적 공유.
        private static readonly HttpClient Http = new HttpClient { Timeout = TimeSpan.FromSeconds(120) };

        /// <summary>
        /// VLM 추론 디바이스 전역 스위치(UI 토글이 설정). Ollama 는 모델을 한 번만 로드하므로 디바이스 선택은
        /// 앱 전역 공통이다(스트림별로 다르게 두면 호출마다 모델 재로딩으로 망가짐).
        /// true = GPU(Ollama 가 VRAM 에 맞게 자동 오프로드), false = CPU 전용(num_gpu=0).
        /// 전환 시 다음 호출에서 Ollama 가 모델을 재로딩한다(콜드 ~1분).
        /// </summary>
        public static bool UseGpu { get; set; } = false;

        private readonly string _endpoint;
        private readonly string _model;

        public OllamaVlmClient(string endpoint = "http://localhost:11434/api/generate", string model = "qwen2.5vl:3b")
        {
            _endpoint = endpoint;
            _model = model;
        }

        public async Task<string> DescribeAsync(byte[] jpeg, string prompt, CancellationToken cancellationToken = default)
        {
            bool useGpu = UseGpu;

            // CPU 전용일 때만 Ollama 서버 우선순위를 BelowNormal 로 낮춰 영상/추론 스레드에 CPU 우선권을 준다
            // (CPU 포화로 영상이 멈추던 문제 완화). GPU 모드면 Normal 로 되돌린다.
            TrySetOllamaPriority(useGpu ? ProcessPriorityClass.Normal : ProcessPriorityClass.BelowNormal);

            var payload = new
            {
                model = _model,
                prompt = prompt,
                images = new[] { Convert.ToBase64String(jpeg) },
                stream = false,
                options = BuildOptions(useGpu),
                keep_alive = "30m"
            };
            using var content = new StringContent(JsonSerializer.Serialize(payload), Encoding.UTF8, "application/json");

            using HttpResponseMessage response = await Http.PostAsync(_endpoint, content, cancellationToken).ConfigureAwait(false);
            response.EnsureSuccessStatusCode();

            string body = await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);
            using JsonDocument doc = JsonDocument.Parse(body);
            if (doc.RootElement.TryGetProperty("response", out JsonElement element))
            {
                return element.GetString()?.Trim() ?? string.Empty;
            }
            return string.Empty;
        }

        /// <summary>
        /// [Phase 3.5] 빈 프롬프트로 /api/generate 를 호출하면 Ollama 가 토큰 생성 없이 모델을 메모리에
        /// 적재만 한다(공식 preload 동작). 첫 실제 묘사 호출의 콜드 로딩(qwen2.5vl:3b)을 앱 시작
        /// 시점으로 당긴다. options/keep_alive 는 실제 호출과 동일하게 맞춰야 같은 적재 상태가 재사용된다
        /// (불일치 시 첫 호출에서 재로딩 발생). 워밍업은 best-effort 라 예외는 호출자가 무시할 수 있다.
        /// </summary>
        public async Task WarmUpAsync(CancellationToken cancellationToken = default)
        {
            bool useGpu = UseGpu;
            TrySetOllamaPriority(useGpu ? ProcessPriorityClass.Normal : ProcessPriorityClass.BelowNormal);

            var payload = new
            {
                model = _model,
                prompt = string.Empty, // 빈 프롬프트 = 적재만 하고 즉시 반환
                stream = false,
                options = BuildOptions(useGpu),
                keep_alive = "30m"
            };
            using var content = new StringContent(JsonSerializer.Serialize(payload), Encoding.UTF8, "application/json");

            using HttpResponseMessage response = await Http.PostAsync(_endpoint, content, cancellationToken).ConfigureAwait(false);
            response.EnsureSuccessStatusCode();
        }

        // DescribeAsync/WarmUpAsync 공통 생성 옵션. 두 경로가 같은 적재 상태를 공유하도록 한 군데서 만든다.
        // num_ctx 2048: 프롬프트 평가/메모리 절감. (keep_alive 는 payload 에서 지정 → 모델 유지, 콜드 재로딩 방지.)
        private static Dictionary<string, object> BuildOptions(bool useGpu)
        {
            var options = new Dictionary<string, object> { ["num_ctx"] = 2048 };
            if (!useGpu)
            {
                options["num_gpu"] = 0;    // CPU 전용: GPU 는 YOLO/영상에 양보
                options["num_thread"] = 4; // 물리코어(8)보다 낮게 → 영상 디코딩용 CPU 여유
            }
            // GPU 모드: num_gpu/num_thread 미지정 → Ollama 가 VRAM 에 맞게 자동 오프로드.
            return options;
        }

        // "ollama" 서버 프로세스(추론 수행)의 우선순위 설정. ("ollama app" GUI 는 대상 아님.)
        // 권한/예외는 무시(같은 사용자 프로세스라 보통 성공).
        private static void TrySetOllamaPriority(ProcessPriorityClass priority)
        {
            try
            {
                foreach (Process p in Process.GetProcessesByName("ollama"))
                {
                    try
                    {
                        if (p.PriorityClass != priority)
                        {
                            p.PriorityClass = priority;
                        }
                    }
                    catch
                    {
                    }
                    finally
                    {
                        p.Dispose();
                    }
                }
            }
            catch
            {
            }
        }
    }
}
