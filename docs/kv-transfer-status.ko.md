# KV transfer PoC 작업 현황

PR #4481의 KV manager 기반(`5e97254e764aa30db4223c33d0dc3b0fded091b8`)에서 WSL x86 CPU로 수행한 작업이다. NPU는 이번 실험 대상이 아니다.

- [후보 조사와 PoC 계획](kv-transfer-poc.md)
- [초기 native CPU 검증](kv-transfer-first-validation.ko.md)
- [논문 방식 Ridge 평가](ridge-paper-evaluation.ko.md)
- [공개 Qwen3 0.6B→1.7B mapper 비교](public-ridge-evaluation.ko.md)
- [작은 평가 결과 JSON](validation/kv-transfer/)
- [mapper 실행 안내](../tools/kv_transfer/README.md), [benchmark 실행 안내](../tools/kv_transfer/benchmark.md)
- [범용 agent 비용 원장 도구](agent-cost-tracking.md), [이번 Ridge 위임 사용량](validation/ridge-paper-agent-cost-20261117.md)

공개 mapper 비교까지 완료했다. Transformers/vLLM과 nntrainer의 동일 입력 KV·logits를 비교하는 추가 오차 진단은 아직 수행하지 않았다.

push 직전 검증: CPU CausalLM build, KV core 64개, mapper/benchmark Python 17개, agent-cost Python 10개 테스트 통과. C/C++ 변경 파일 clang-format 14 적용. 전체 플랫폼 CI 검증은 수행하지 않았다.

대용량 모델·KV snapshot·mapper 가중치, 개인 rollout 및 로컬 usage 원장은 Git에 포함하지 않는다. 재현 명령과 데이터/모델 revision은 각 보고서에 기록했다.
