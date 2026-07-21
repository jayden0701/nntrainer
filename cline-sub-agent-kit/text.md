폴더 구조와 문서를 먼저 파악하겠습니다.

분석이 완료됐습니다. 구조와 당신의 환경에 맞는 사용법을 설명드립니다.

---

## 키트 구조 이해

이 키트는 3개 컴포넌트로 구성됩니다:

```
Claude Code (main agent)
    └─ MCP tool: cline_sub_agent
           └─ server.py (MCP 서버, stdio)
                  └─ HTTP localhost:39111
                         └─ claude-cline-bridge (VS Code 확장)
                                └─ Cline (sub-agent 실행)
```

핵심 제약: **Claude CLI · VS Code 브릿지 · MCP server.py가 반드시 같은 머신에 있어야 함** (브릿지가 `127.0.0.1` 전용)

---

## 당신의 환경 (VS Code SSH) 에서의 적용

**좋은 소식:** VS Code SSH 확장으로 접속 시, VS Code 확장들은 **리모트 Linux 머신에서 실행**됩니다. 즉, 세 컴포넌트가 모두 Linux에 올라가므로 localhost 제약이 문제 없습니다.

```
[Windows PC]              [Linux 개발 서버]
  VS Code UI  ──SSH──>  VS Code Server
                         └─ bridge 확장 (39111 listen)
                         └─ Cline 확장
                         └─ Claude CLI
                         └─ mcp/server.py
```

---

## 설치 순서

**Step 1. Linux 서버에 VSIX 파일 전송**

현재 VSIX가 Windows `Downloads`에 있으니 SCP 등으로 Linux로 복사합니다:

```powershell
scp "C:\Users\jrock.oh\Downloads\cline-sub-agent-kit\cline-sub-agent-kit\claude-cline-bridge.claude-cline-bridge-0.0.3.vsix" <user>@<linux-host>:~/
```

`mcp/server.py`도 함께 복사합니다:

```powershell
scp -r "C:\Users\jrock.oh\Downloads\cline-sub-agent-kit\cline-sub-agent-kit\mcp" <user>@<linux-host>:~/cline-sub-agent-kit/
```

**Step 2. VS Code에서 VSIX 설치 (Remote SSH 세션에서)**

VS Code가 SSH로 Linux에 연결된 상태에서:
Extensions 패널 → `⋯` → **Install from VSIX…** → Linux 홈 디렉토리의 `.vsix` 선택 → Reload Window

설치 확인 (Linux 터미널에서):
```bash
curl -s localhost:39111/health
# {"ok":true,"clineActivated":true,...} 이면 성공
```

**Step 3. MCP 서버 등록 (Linux 터미널에서)**

```bash
claude mcp add cline-sub-agent -s user -- python3 ~/cline-sub-agent-kit/mcp/server.py
```

Claude Code 재시작 후 `/mcp` 명령으로 `cline-sub-agent` 도구 확인.

---

## 사용 전 체크리스트 (매번)

| 항목 | 이유 |
|------|------|
| VS Code 창 하나만 열기 | Cline 인스턴스 충돌 방지 |
| 폴더 열어둘 것 (File → Open Folder) | Cline이 작업할 workspace 필요 |
| Cline을 **Act 모드**로 설정 | Ask/Plan 모드면 파일 수정 불가 |

---

## 활용 방법 (홍승백님 설명 기준)

Claude Code가 작업 중 sub-agent가 필요하다 판단할 때, 기본은 Haiku/Sonnet/Opus 공식 sub-agent를 쓰는데, **명시적으로 Cline을 쓰라고 지시**하면 됩니다. 예시 프롬프트:

```
탐색이나 구현 작업이 필요할 때는 공식 sub-agent 대신 
cline_sub_agent 도구를 사용해서 Cline에게 위임해줘.
```

읽기 전용 작업(탐색/리뷰)은 task 문자열에 `"읽고 보고만, 수정 금지"` 명시하면 `changedFiles`가 비어서 돌아옵니다.

---

**한 줄 요약:** VS Code SSH 환경이면 세 컴포넌트 모두 Linux에 올라가므로 localhost 제약 문제없이 동작합니다. VSIX와 server.py를 Linux로 옮기고, Remote SSH 세션에서 확장 설치 → MCP 등록 → 재시작 순서로 진행하면 됩니다.