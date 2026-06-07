using System.Threading;
using System.Threading.Tasks;

namespace Vision.MultiStream.Inference.Services.Vlm
{
    /// <summary>
    /// 비전-언어 모델(VLM) 호출 경계. "JPEG + 프롬프트 → 묘사 텍스트".
    /// 백엔드(Ollama LLaVA → llama.cpp → 직접 ONNX)를 갈아끼울 수 있게 이 한 군데만 인터페이스로 둔다.
    /// </summary>
    public interface IVlmClient
    {
        Task<string> DescribeAsync(byte[] jpeg, string prompt, CancellationToken cancellationToken = default);
    }
}
