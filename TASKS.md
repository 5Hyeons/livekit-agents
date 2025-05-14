# STF (Speech-To-Face) 애니메이션 데이터 스트리밍 구현 작업

## 배경
현재 STF(Speech-To-Face) 구현은 애니메이션 데이터(블렌드쉐입)를 시각화를 위한 비디오 프레임으로 변환합니다. 이제는 원시 애니메이션 데이터를 직접 Unity로 스트리밍하여 Unity가 렌더링 과정을 처리하도록 시스템을 수정해야 합니다.

## 구현 작업

### 1. 새로운 애니메이션 데이터 스트림 인터페이스 추가
- [x] `VideoOutput`과 유사하게 `ABC`를 상속받는 새로운 `AnimationDataOutput` 클래스를 `livekit/agents/voice/io.py`에 생성
- [x] 애니메이션 데이터를 위한 `capture_frame` 및 `flush` 추상 메서드 정의
- [x] `AgentOutput` 클래스에 애니메이션 데이터 출력 속성 추가

### 2. 애니메이션 데이터용 ByteStream 구현
- [x] 새 파일 `livekit/agents/voice/avatar/_animation_datastream_io.py` 생성
- [x] `DataStreamAudioOutput`을 모델로 한 `ByteStreamAnimationOutput` 클래스 구현
- [x] 애니메이션 데이터(NumPy 배열)의 바이트 직렬화 설정
- [x] 적절한 메타데이터와 함께 바이트 스트림 쓰기 구현
- [x] 애니메이션 데이터 스트림용 토픽 식별자 정의(예: "lk.animation_stream")

### 3. STF, STFStream 클래스 업데이트 및 버그 수정
- [x] `livekit/agents/stf/stf.py`의 `STF` 프로토콜을 수정하여 원시 애니메이션 데이터 출력을 위한 새 메서드 포함
- [x] `STFStream` 클래스를 업데이트하여 비디오 프레임으로 변환하는 대신 원시 애니메이션 데이터 출력
- [x] `__anext__` 메서드를 변경하여 `VideoFrame` 대신 애니메이션 데이터 반환
- [x] 블렌드쉐입 데이터의 바이너리 직렬화 구현
- [x] `__anext__` 메서드에서 `NoneType` 오류 수정 - `self._task._result` 안전한 접근 로직 추가
- [x] Numba 로깅 레벨 조정 - 과도한 디버그 메시지 감소

### 4. `stf_node` 구현 수정
- [x] `Agent` 클래스의 `stf_node` 시그니처를 업데이트하여 애니메이션 데이터 반환
- [x] 새 애니메이션 데이터 인터페이스를 사용하도록 `default.stf_node` 구현 수정

### 5. 생성 모듈의 파이프라인 통합 업데이트
- [x] `livekit/agents/voice/generation.py`에 `_AnimationOutput` 클래스 생성
- [x] 애니메이션 데이터용 `perform_animation_forwarding` 함수 구현
- [x] 애니메이션 데이터를 처리하도록 `_pipeline_reply_task` 및 `_realtime_generation_task` 업데이트

### 6. `perform_stf_inference` 함수 수정
- [x] 비디오 대신 애니메이션 데이터 채널을 생성하도록 함수 업데이트
- [x] 비디오 프레임 변환 기능 제거
- [x] NumPy 애니메이션 데이터의 적절한 바이너리 직렬화 구현

### 7. Room IO 구현 업데이트
- [x] `room_io.py`의 `RoomOutputOptions`에 애니메이션 데이터 지원 추가
- [x] `room_io.py`에 애니메이션 출력 핸들러 생성
- [x] `RoomIO.start()`에서 ByteStream 초기화 구현

### 8. 샘플 애플리케이션 및 테스트
- [x] 새로운 애니메이션 데이터 스트리밍을 사용하도록 `face_animation_agent.py` 업데이트
- [x] 애니메이션 데이터 출력을 위한 구성 옵션 추가
- [ ] 데이터 수신을 확인하기 위한 간단한 Unity 테스트 씬 생성

### 9. Unity 클라이언트 구현(Unity 측)
- [ ] 애니메이션 데이터를 수신하기 위한 ByteStreamReader 구현 개발
- [ ] 바이트에서 애니메이션 데이터의 역직렬화 구현
- [ ] 3D 얼굴 모델에 애니메이션 데이터를 적용하는 컴포넌트 생성
- [ ] Unity 클라이언트에서 스트림 연결 및 데이터 구독 설정

### 10. 문서화 및 정리
- [x] 모든 새 클래스 및 메서드에 문서 추가
- [x] 비디오 기반 시각화 코드 정리 또는 폐기
- [x] `face_animation_agent.py` 문서 업데이트
- [ ] 새로운 워크플로우를 보여주는 샘플 생성

## 업데이트 및 문제 해결

### 애니메이션 데이터 전송 문제 해결 (2025-04-29)
- [x] `_participant_identity`가 None인 문제 발견 - `RoomIO` 객체를 올바르게 구성하지 않아 발생
- [x] `RoomOutputOptions`에 직접 `participant` 매개변수를 추가하려고 했으나 지원되지 않음 확인
- [x] 코드 수정: `RoomIO` 객체 생성 시 `participant` 매개변수를 전달하도록 `face_animation_agent.py` 변경
- [x] `room_io.start()`와 `session.start(agent)`를 분리하여 올바른 초기화 순서 적용
- [x] 애니메이션 데이터가 지정된 참가자에게 성공적으로 전송되는 것을 확인

## 우선 순위 항목
1. ✅ 원시 애니메이션 데이터를 출력하도록 `STF` 프로토콜 및 `STFStream` 업데이트
2. ✅ 애니메이션 데이터 전송을 위한 `ByteStreamAnimationOutput` 구현
3. ✅ 에이전트 파이프라인의 통합 업데이트
4. ⏳ Unity 측 애니메이션 데이터 수신 컴포넌트 생성
