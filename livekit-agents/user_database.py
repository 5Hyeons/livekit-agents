import sqlite3
import json
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
import uuid
import os
import re
from contextlib import contextmanager

logger = logging.getLogger("user-database")

def sanitize_filename(filename: str) -> str:
    """
    파일명에서 특수문자를 제거하고 안전한 파일명으로 변환
    경로 traversal 공격 방지를 위해 '../' 등의 패턴 제거
    """
    # 경로 traversal 패턴 제거
    filename = filename.replace('..', '').replace('/', '').replace('\\', '')
    # 특수문자를 언더스코어로 대체 (영문자, 숫자, 하이픈, 언더스코어만 허용)
    filename = re.sub(r'[^a-zA-Z0-9_-]', '_', filename)
    # 연속된 언더스코어를 하나로 줄임
    filename = re.sub(r'_+', '_', filename)
    # 앞뒤 언더스코어 제거
    filename = filename.strip('_')
    # 빈 문자열이면 기본값 사용
    if not filename:
        filename = 'unknown_user'
    return filename

@dataclass
class UserData:
    """사용자 정보를 저장하는 데이터 클래스"""
    participant_id: str
    display_name: Optional[str] = None
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    language: str = "ko"  # 사용자 언어 (기본값: 한국어)
    
@dataclass
class ChatMessage:
    """채팅 메시지를 저장하는 데이터 클래스"""
    participant_id: str
    session_id: str
    timestamp: datetime
    role: str  # 'user' or 'assistant'
    content: str
    interrupted: bool = False

class UserDatabase:
    """사용자 정보와 채팅 기록을 관리하는 데이터베이스"""
    
    def __init__(self, participant_id: str, db_dir: str = "data"):
        # 사용자 ID를 안전한 파일명으로 변환
        safe_id = sanitize_filename(participant_id)
        self.participant_id = participant_id
        self.db_path = os.path.join(db_dir, f"users_{safe_id}.db")
        
        # 데이터 디렉토리 생성
        os.makedirs(db_dir, exist_ok=True)
        self._init_database()
        
        logger.info(f"사용자 {participant_id}용 DB 초기화: {self.db_path}")
        
    @contextmanager
    def get_connection(self):
        """데이터베이스 연결 컨텍스트 매니저"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
            
    def _init_database(self):
        """데이터베이스 테이블 초기화"""
        with self.get_connection() as conn:
            # 사용자 테이블
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    participant_id TEXT PRIMARY KEY,
                    display_name TEXT,
                    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    language TEXT DEFAULT 'ko',
                    metadata TEXT,
                    remaining_tokens INTEGER DEFAULT 2000,
                    total_tokens_granted INTEGER DEFAULT 2000,
                    total_tokens_used INTEGER DEFAULT 0
                )
            """)
            
            # 채팅 기록 테이블
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    participant_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    interrupted BOOLEAN DEFAULT FALSE,
                    FOREIGN KEY (participant_id) REFERENCES users(participant_id)
                )
            """)
            
            # 사용량 메트릭 테이블
            conn.execute("""
                CREATE TABLE IF NOT EXISTS usage_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    participant_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    metric_type TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    request_id TEXT,
                    
                    -- LLM specific fields
                    prompt_tokens INTEGER,
                    prompt_cached_tokens INTEGER,
                    completion_tokens INTEGER,
                    total_tokens INTEGER,
                    
                    -- TTS specific fields
                    characters_count INTEGER,
                    audio_duration REAL,
                    
                    -- Common fields
                    duration REAL,
                    cancelled BOOLEAN DEFAULT FALSE,
                    
                    -- Additional metadata
                    metadata TEXT,
                    
                    FOREIGN KEY (participant_id) REFERENCES users(participant_id)
                )
            """)
            
            # 인덱스 생성
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_participant ON chat_history(participant_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_session ON chat_history(session_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_participant ON usage_metrics(participant_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_session ON usage_metrics(session_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_type ON usage_metrics(metric_type)")
            
            # 기존 users 테이블에 토큰 컬럼 추가 (마이그레이션)
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(users)")
            columns = [column[1] for column in cursor.fetchall()]
            
            if 'remaining_tokens' not in columns:
                conn.execute("ALTER TABLE users ADD COLUMN remaining_tokens INTEGER DEFAULT 2000")
                conn.execute("UPDATE users SET remaining_tokens = 2000 WHERE remaining_tokens IS NULL")
                logger.info("Added remaining_tokens column to users table")
            
            if 'total_tokens_granted' not in columns:
                conn.execute("ALTER TABLE users ADD COLUMN total_tokens_granted INTEGER DEFAULT 2000")
                conn.execute("UPDATE users SET total_tokens_granted = 2000 WHERE total_tokens_granted IS NULL")
                logger.info("Added total_tokens_granted column to users table")
            
            if 'total_tokens_used' not in columns:
                conn.execute("ALTER TABLE users ADD COLUMN total_tokens_used INTEGER DEFAULT 0")
                conn.execute("UPDATE users SET total_tokens_used = 0 WHERE total_tokens_used IS NULL")
                logger.info("Added total_tokens_used column to users table")
            
    def get_or_create_user(self, participant_id: str) -> UserData:
        """사용자 정보를 가져오거나 새로 생성"""
        with self.get_connection() as conn:
            # 기존 사용자 조회
            result = conn.execute(
                "SELECT * FROM users WHERE participant_id = ?", 
                (participant_id,)
            ).fetchone()
            
            if result:
                # 기존 사용자
                user_data = UserData(
                    participant_id=result['participant_id'],
                    display_name=result['display_name'],
                    first_seen=datetime.fromisoformat(result['first_seen']),
                    last_seen=datetime.fromisoformat(result['last_seen']),
                    language=result['language'] or 'ko'
                )
                logger.info(f"기존 사용자 로드: {participant_id}, 이름: {user_data.display_name}, 언어: {user_data.language}")
            else:
                # 새 사용자 생성
                now = datetime.now()
                conn.execute(
                    "INSERT INTO users (participant_id, first_seen, last_seen, language, remaining_tokens, total_tokens_granted, total_tokens_used) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (participant_id, now, now, 'ko', 2000, 2000, 0)
                )
                user_data = UserData(
                    participant_id=participant_id,
                    first_seen=now,
                    last_seen=now,
                    language='ko'
                )
                logger.info(f"새 사용자 생성: {participant_id} (2000 토큰 부여)")
                
        return user_data
        
    def update_user_name(self, participant_id: str, display_name: str):
        """사용자 이름 업데이트"""
        with self.get_connection() as conn:
            conn.execute(
                "UPDATE users SET display_name = ?, last_seen = ? WHERE participant_id = ?",
                (display_name, datetime.now(), participant_id)
            )
            logger.info(f"사용자 이름 업데이트: {participant_id} -> {display_name}")
            
    def update_user_language(self, participant_id: str, language: str):
        """사용자 언어 업데이트"""
        with self.get_connection() as conn:
            conn.execute(
                "UPDATE users SET language = ?, last_seen = ? WHERE participant_id = ?",
                (language, datetime.now(), participant_id)
            )
            logger.info(f"사용자 언어 업데이트: {participant_id} -> {language}")
            
    def update_user_metadata(self, participant_id: str, metadata: Dict[str, Any]):
        """사용자 메타데이터 업데이트"""
        with self.get_connection() as conn:
            metadata_json = json.dumps(metadata) if metadata else None
            conn.execute(
                "UPDATE users SET metadata = ?, last_seen = ? WHERE participant_id = ?",
                (metadata_json, datetime.now(), participant_id)
            )
            logger.info(f"사용자 메타데이터 업데이트: {participant_id}")
            
    def get_user_metadata(self, participant_id: str) -> Optional[Dict[str, Any]]:
        """사용자 메타데이터 조회"""
        with self.get_connection() as conn:
            result = conn.execute(
                "SELECT metadata FROM users WHERE participant_id = ?",
                (participant_id,)
            ).fetchone()
            
            if result and result['metadata']:
                return json.loads(result['metadata'])
            return None
            
    def update_last_seen(self, participant_id: str):
        """마지막 접속 시간 업데이트"""
        with self.get_connection() as conn:
            conn.execute(
                "UPDATE users SET last_seen = ? WHERE participant_id = ?",
                (datetime.now(), participant_id)
            )
            
            
    def save_chat_message(self, message: ChatMessage):
        """채팅 메시지 저장"""
        with self.get_connection() as conn:
            conn.execute(
                """INSERT INTO chat_history 
                   (participant_id, session_id, timestamp, role, content, interrupted) 
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (message.participant_id, message.session_id, message.timestamp, 
                 message.role, message.content, message.interrupted)
            )
            
    def save_chat_messages(self, messages: List[ChatMessage]):
        """여러 채팅 메시지 일괄 저장"""
        with self.get_connection() as conn:
            conn.executemany(
                """INSERT INTO chat_history 
                   (participant_id, session_id, timestamp, role, content, interrupted) 
                   VALUES (?, ?, ?, ?, ?, ?)""",
                [(m.participant_id, m.session_id, m.timestamp, 
                  m.role, m.content, m.interrupted) for m in messages]
            )
            logger.info(f"{len(messages)}개의 메시지 저장 완료")
            
    def get_chat_history(self, participant_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """사용자의 최근 채팅 기록 조회"""
        with self.get_connection() as conn:
            results = conn.execute(
                """SELECT * FROM chat_history 
                   WHERE participant_id = ? 
                   ORDER BY timestamp DESC 
                   LIMIT ?""",
                (participant_id, limit)
            ).fetchall()
            
            # 시간 순서로 정렬 (오래된 것부터)
            messages = []
            for row in reversed(results):
                messages.append({
                    'session_id': row['session_id'],
                    'timestamp': row['timestamp'],
                    'role': row['role'],
                    'content': row['content'],
                    'interrupted': bool(row['interrupted'])
                })
                
            return messages
            
    def get_recent_context(self, participant_id: str, message_count: int = 10) -> str:
        """최근 대화 컨텍스트를 문자열로 반환"""
        messages = self.get_chat_history(participant_id, message_count)
        
        if not messages:
            return ""
            
        context_lines = []
        current_session = None
        
        for msg in messages:
            # 세션이 바뀌면 구분선 추가
            if current_session != msg['session_id']:
                if current_session is not None:
                    context_lines.append("---")
                current_session = msg['session_id']
                
            role_name = "사용자" if msg['role'] == "user" else "어시스턴트"
            context_lines.append(f"{msg['role']}: {msg['content']}")
            
        return "\n".join(context_lines)
        
    def get_user_summary(self, participant_id: str) -> Dict[str, Any]:
        """사용자 요약 정보 반환"""
        user_data = self.get_or_create_user(participant_id)
        chat_count = 0
        
        with self.get_connection() as conn:
            result = conn.execute(
                "SELECT COUNT(*) as count FROM chat_history WHERE participant_id = ?",
                (participant_id,)
            ).fetchone()
            chat_count = result['count']
            
        return {
            'participant_id': participant_id,
            'display_name': user_data.display_name,
            'first_seen': user_data.first_seen.isoformat() if user_data.first_seen else None,
            'last_seen': user_data.last_seen.isoformat() if user_data.last_seen else None,
            'total_messages': chat_count
        }
    
    def clear_chat_history(self) -> int:
        """현재 participant의 모든 채팅 기록 삭제"""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM chat_history WHERE participant_id = ?",
                (self.participant_id,)
            )
            deleted_count = cursor.rowcount
            logger.info(f"사용자 {self.participant_id}의 채팅 기록 {deleted_count}개 삭제")
            return deleted_count
    
    def save_llm_usage(self, participant_id: str, session_id: str, metrics: Dict[str, Any]):
        """LLM 사용량 저장"""
        with self.get_connection() as conn:
            conn.execute(
                """INSERT INTO usage_metrics 
                   (participant_id, session_id, metric_type, timestamp, request_id,
                    prompt_tokens, prompt_cached_tokens, completion_tokens, total_tokens,
                    duration, cancelled, metadata) 
                   VALUES (?, ?, 'llm', ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    participant_id,
                    session_id,
                    datetime.now(),
                    metrics.get('request_id'),
                    metrics.get('prompt_tokens', 0),
                    metrics.get('prompt_cached_tokens', 0),
                    metrics.get('completion_tokens', 0),
                    metrics.get('total_tokens', 0),
                    metrics.get('duration', 0.0),
                    metrics.get('cancelled', False),
                    json.dumps({
                        'label': metrics.get('label'),
                        'ttft': metrics.get('ttft'),
                        'tokens_per_second': metrics.get('tokens_per_second'),
                        'speech_id': metrics.get('speech_id')
                    })
                )
            )
            logger.debug(f"LLM usage saved: {participant_id}, tokens: {metrics.get('total_tokens')}")
    
    def save_tts_usage(self, participant_id: str, session_id: str, metrics: Dict[str, Any]):
        """TTS 사용량 저장 및 토큰 차감"""
        characters_count = metrics.get('characters_count', 0)
        
        with self.get_connection() as conn:
            # TTS 사용량 저장
            conn.execute(
                """INSERT INTO usage_metrics 
                   (participant_id, session_id, metric_type, timestamp, request_id,
                    characters_count, audio_duration, duration, cancelled, metadata) 
                   VALUES (?, ?, 'tts', ?, ?, ?, ?, ?, ?, ?)""",
                (
                    participant_id,
                    session_id,
                    datetime.now(),
                    metrics.get('request_id'),
                    characters_count,
                    metrics.get('audio_duration', 0.0),
                    metrics.get('duration', 0.0),
                    metrics.get('cancelled', False),
                    json.dumps({
                        'label': metrics.get('label'),
                        'ttfb': metrics.get('ttfb'),
                        'streamed': metrics.get('streamed'),
                        'segment_id': metrics.get('segment_id'),
                        'speech_id': metrics.get('speech_id')
                    })
                )
            )
            logger.debug(f"TTS usage saved: {participant_id}, chars: {characters_count}")
            
        # 토큰 차감 (1 character = 1 token)
        if characters_count > 0:
            success = self.deduct_tokens(participant_id, characters_count)
            if not success:
                logger.warning(f"Token deduction failed for {participant_id}: insufficient balance for {characters_count} tokens")
    
    def get_usage_summary(self, participant_id: str) -> Dict[str, Any]:
        """사용자의 전체 사용량 요약 조회"""
        with self.get_connection() as conn:
            # LLM 사용량 합계
            llm_result = conn.execute(
                """SELECT 
                    COUNT(*) as request_count,
                    SUM(prompt_tokens) as total_prompt_tokens,
                    SUM(prompt_cached_tokens) as total_cached_tokens,
                    SUM(completion_tokens) as total_completion_tokens,
                    SUM(total_tokens) as total_tokens
                   FROM usage_metrics 
                   WHERE participant_id = ? AND metric_type = 'llm'""",
                (participant_id,)
            ).fetchone()
            
            # TTS 사용량 합계
            tts_result = conn.execute(
                """SELECT 
                    COUNT(*) as request_count,
                    SUM(characters_count) as total_characters,
                    SUM(audio_duration) as total_audio_duration
                   FROM usage_metrics 
                   WHERE participant_id = ? AND metric_type = 'tts'""",
                (participant_id,)
            ).fetchone()
            
            return {
                'participant_id': participant_id,
                'llm': {
                    'request_count': llm_result['request_count'] or 0,
                    'total_prompt_tokens': llm_result['total_prompt_tokens'] or 0,
                    'total_cached_tokens': llm_result['total_cached_tokens'] or 0,
                    'total_completion_tokens': llm_result['total_completion_tokens'] or 0,
                    'total_tokens': llm_result['total_tokens'] or 0
                },
                'tts': {
                    'request_count': tts_result['request_count'] or 0,
                    'total_characters': tts_result['total_characters'] or 0,
                    'total_audio_duration': tts_result['total_audio_duration'] or 0.0
                }
            }
    
    def get_remaining_tokens(self, participant_id: str) -> int:
        """사용자의 남은 토큰 조회"""
        with self.get_connection() as conn:
            result = conn.execute(
                "SELECT remaining_tokens FROM users WHERE participant_id = ?",
                (participant_id,)
            ).fetchone()
            
            if result:
                return result['remaining_tokens'] or 0
            return 0
    
    def deduct_tokens(self, participant_id: str, amount: int) -> bool:
        """토큰 차감 (잔액 부족시 False 반환)"""
        with self.get_connection() as conn:
            # 현재 잔액 확인
            result = conn.execute(
                "SELECT remaining_tokens FROM users WHERE participant_id = ?",
                (participant_id,)
            ).fetchone()
            
            if not result:
                logger.error(f"User not found: {participant_id}")
                return False
            
            current_tokens = result['remaining_tokens'] or 0
            
            if current_tokens < amount:
                logger.warning(f"Insufficient tokens for {participant_id}: {current_tokens} < {amount}")
                # 잔액 부족이어도 0으로 만들고 진행 (음수 방지)
                conn.execute(
                    """UPDATE users 
                       SET remaining_tokens = 0,
                           total_tokens_used = total_tokens_used + ?,
                           last_seen = ?
                       WHERE participant_id = ?""",
                    (current_tokens, datetime.now(), participant_id)
                )
                return False
            
            # 토큰 차감
            conn.execute(
                """UPDATE users 
                   SET remaining_tokens = remaining_tokens - ?,
                       total_tokens_used = total_tokens_used + ?,
                       last_seen = ?
                   WHERE participant_id = ?""",
                (amount, amount, datetime.now(), participant_id)
            )
            
            new_balance = current_tokens - amount
            logger.debug(f"Tokens deducted for {participant_id}: {amount} (remaining: {new_balance})")
            return True
    
    def add_tokens(self, participant_id: str, amount: int):
        """토큰 충전 (관리자용)"""
        with self.get_connection() as conn:
            conn.execute(
                """UPDATE users 
                   SET remaining_tokens = remaining_tokens + ?,
                       total_tokens_granted = total_tokens_granted + ?,
                       last_seen = ?
                   WHERE participant_id = ?""",
                (amount, amount, datetime.now(), participant_id)
            )
            
            # 새 잔액 조회
            result = conn.execute(
                "SELECT remaining_tokens FROM users WHERE participant_id = ?",
                (participant_id,)
            ).fetchone()
            
            if result:
                new_balance = result['remaining_tokens']
                logger.info(f"Tokens added for {participant_id}: +{amount} (new balance: {new_balance})")
    
    def get_token_info(self, participant_id: str) -> Dict[str, int]:
        """사용자의 토큰 정보 조회"""
        with self.get_connection() as conn:
            result = conn.execute(
                """SELECT remaining_tokens, total_tokens_granted, total_tokens_used 
                   FROM users WHERE participant_id = ?""",
                (participant_id,)
            ).fetchone()
            
            if result:
                return {
                    'remaining_tokens': result['remaining_tokens'] or 0,
                    'total_tokens_granted': result['total_tokens_granted'] or 2000,
                    'total_tokens_used': result['total_tokens_used'] or 0
                }
            return {
                'remaining_tokens': 0,
                'total_tokens_granted': 0,
                'total_tokens_used': 0
            }
    
    def get_session_usage(self, session_id: str) -> Dict[str, Any]:
        """특정 세션의 사용량 조회"""
        with self.get_connection() as conn:
            results = conn.execute(
                """SELECT * FROM usage_metrics 
                   WHERE session_id = ? 
                   ORDER BY timestamp""",
                (session_id,)
            ).fetchall()
            
            llm_usage = []
            tts_usage = []
            
            for row in results:
                record = {
                    'timestamp': row['timestamp'],
                    'request_id': row['request_id'],
                    'duration': row['duration'],
                    'cancelled': bool(row['cancelled'])
                }
                
                if row['metric_type'] == 'llm':
                    record.update({
                        'prompt_tokens': row['prompt_tokens'],
                        'cached_tokens': row['prompt_cached_tokens'],
                        'completion_tokens': row['completion_tokens'],
                        'total_tokens': row['total_tokens']
                    })
                    llm_usage.append(record)
                elif row['metric_type'] == 'tts':
                    record.update({
                        'characters_count': row['characters_count'],
                        'audio_duration': row['audio_duration']
                    })
                    tts_usage.append(record)
            
            return {
                'session_id': session_id,
                'llm_usage': llm_usage,
                'tts_usage': tts_usage
            }