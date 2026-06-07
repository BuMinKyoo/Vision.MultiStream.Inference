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
    /// 로컬 Ollama 서버의 LLaVA 계열 모델을 HTTP 로 호출하는 <see cref="IVlmClient"/> 구현.
    /// 사전 준비(코드 외): 터미널에서 `ollama pull llava` 후 Ollama 데몬 기동(기본 11434 포트).
    ///
    /// 요청 (POST /api/generate): { "model":"llava", "prompt":"...", "images":["&lt;base64 JPEG&gt;"], "stream":false }
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
        public static bool UseGpu { get; set; } = true;

        private readonly string _endpoint;
        private readonly string _model;

        public OllamaVlmClient(string endpoint = "http://localhost:11434/api/generate", string model = "llava")
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

            // num_ctx 2048: 프롬프트 평가/메모리 절감. keep_alive: 모델을 유지해 콜드 재로딩 방지.
            var options = new Dictionary<string, object> { ["num_ctx"] = 2048 };
            if (!useGpu)
            {
                options["num_gpu"] = 0;    // CPU 전용: GPU 는 YOLO/영상에 양보
                options["num_thread"] = 4; // 물리코어(8)보다 낮게 → 영상 디코딩용 CPU 여유
            }
            // GPU 모드: num_gpu/num_thread 미지정 → Ollama 가 VRAM 에 맞게 자동 오프로드.

            var payload = new
            {
                model = _model,
                prompt = prompt,
                images = new[] { Convert.ToBase64String(jpeg) },
                stream = false,
                options = options,
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
