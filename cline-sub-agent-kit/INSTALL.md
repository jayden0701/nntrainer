# cline_sub_agent — Cline을 Claude Code의 서브에이전트로

Claude Code(또는 다른 오케스트레이터)가 **Cline을 범용 서브에이전트로 부르는** MCP 도구.
`task`(자연어)를 넘기면 Cline이 열린 워크스페이스에서 그 일을 수행하고(도구 승인 자동 처리)
**최종 메시지 + 변경 파일 목록**을 돌려준다. 탐색·구현·리팩토링·버그수정·리뷰·빌드/테스트 등
**무엇에든** 쓸 수 있고, **어떻게 활용할지는 각자 자유**다(이 문서는 도구 설치까지만 다룬다).

구성: VS Code 확장(브릿지) 1개 + MCP 서버(`server.py`) 1개. 브릿지는 `127.0.0.1`만 사용.

## 전제
- VS Code + **Cline 확장** 설치 (사내 `cline-sr` 포크든 공개 Cline이든 자동 감지)
- **Claude Code**, **python3**
- Claude Code · VS Code(Cline) · 브릿지가 **같은 머신**에 있어야 함 (브릿지가 localhost 전용)

## 1. 브릿지 확장 설치
VS Code → Extensions 패널 → 우상단 `⋯` → **Install from VSIX…** →
`claude-cline-bridge.claude-cline-bridge-0.0.3.vsix` 선택 → **Developer: Reload Window**.

확인:
```
curl -s localhost:39111/health
```
`{"ok":true,"clineActivated":true,...}` 이면 성공.
(39111이 점유돼 있으면 39112~39130으로 폴백하며, MCP 서버가 알아서 찾는다.)

## 2. MCP 도구 등록
```
claude mcp add cline-sub-agent -s user -- python3 <이폴더경로>/mcp/server.py
```
→ **Claude Code 재시작** (MCP는 시작 시 로드되므로 재시작해야 `cline_sub_agent` 가 도구
목록에 뜬다). 확인: Claude Code에서 `/mcp` 에 `cline-sub-agent` 표시.

## 3. 사용 전 체크 (매번 — 셋 다 중요)
- **VS Code 창은 하나만.** 창이 여러 개면 Cline 인스턴스가 충돌해 파일이 "수정됨 ●"인 채
  저장이 안 되고 호출이 멈춘다.
- **폴더를 열어둘 것** (File → Open Folder). 폴더가 없으면 `health` 의 `workspace` 가
  `null` 이고 Cline이 파일을 만들 곳이 없다.
- Cline을 **Act 모드**로 (사이드바 Plan/Act/Ask 토글). Ask면 파일을 못 만든다.

## 사용
`cline_sub_agent` 도구에 `task`(자연어)를 주면 Cline이 수행하고
`{ finalMessage, changedFiles }` 를 돌려준다. 읽기전용으로 쓰려면 task에 "읽고 보고만,
수정 금지"처럼 명시하면 된다(그러면 changedFiles는 비어서 온다). 한 번에 한 task
(호출은 직렬화 — Cline은 한 작업씩). 어떤 워크플로로 엮을지는 사용자 자유.

## 트러블슈팅
- **Cline 감지 안 됨** (`clineActivated:false` 또는 `clineId` null): 설치된 Cline 확장 id를
  확인해(Extensions에서 우클릭 → Copy Extension ID) `cline-sr.cline-sr` /
  `saoudrizwan.claude-dev` 가 아니면 환경변수 `CLAUDE_CLINE_BRIDGE_CLINE_ID=<id>` 로 지정.
- **파일이 "수정됨 ●"인 채 멈춤**: VS Code 창이 여러 개다 → 하나만 남기고 Reload Window.
- **`cline_sub_agent` 도구가 안 보임**: Claude Code 재시작 필요.
- **토큰**: 브릿지 최초 기동 시 `~/.claude-cline-bridge-token` 자동 생성(MCP 서버가 같은
  파일을 읽으므로 보통 신경 쓸 필요 없음).
