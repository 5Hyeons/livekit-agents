import sqlite3
import json
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
import uuid
import os
from contextlib import contextmanager

logger = logging.getLogger("user-database")

@dataclass
class UserData:
    """사용자 정보를 저장하는 데이터 클래스"""
    participant_id: str
    display_name: Optional[str] = None
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
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
    
    def __init__(self, db_path: str = "data/users.db"):
        self.db_path = db_path
        # 데이터 디렉토리 생성
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_database()
        
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
                    metadata TEXT
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
            
            # 인덱스 생성
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_participant ON chat_history(participant_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_session ON chat_history(session_id)")
            
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
                    last_seen=datetime.fromisoformat(result['last_seen'])
                )
                logger.info(f"기존 사용자 로드: {participant_id}, 이름: {user_data.display_name}")
            else:
                # 새 사용자 생성
                now = datetime.now()
                conn.execute(
                    "INSERT INTO users (participant_id, first_seen, last_seen) VALUES (?, ?, ?)",
                    (participant_id, now, now)
                )
                user_data = UserData(
                    participant_id=participant_id,
                    first_seen=now,
                    last_seen=now
                )
                logger.info(f"새 사용자 생성: {participant_id}")
                
        return user_data
        
    def update_user_name(self, participant_id: str, display_name: str):
        """사용자 이름 업데이트"""
        with self.get_connection() as conn:
            conn.execute(
                "UPDATE users SET display_name = ?, last_seen = ? WHERE participant_id = ?",
                (display_name, datetime.now(), participant_id)
            )
            logger.info(f"사용자 이름 업데이트: {participant_id} -> {display_name}")
            
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
            context_lines.append(f"{role_name}: {msg['content']}")
            
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