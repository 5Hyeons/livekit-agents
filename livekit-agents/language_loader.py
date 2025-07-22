import json
import logging
import os
import random
from typing import Dict, List, Optional
from functools import lru_cache

logger = logging.getLogger("language-loader")

# 지원하는 언어 목록
SUPPORTED_LANGUAGES = ["ko", "en", "ja", "zh"]

class LanguageConfig:
    """언어별 설정을 저장하는 클래스"""
    
    def __init__(self, language_code: str, config_data: Dict):
        self.language_code = language_code
        
        # base_instructions가 리스트일 경우 줄바꿈으로 합치기
        base_instructions_raw = config_data.get("base_instructions", "")
        if isinstance(base_instructions_raw, list):
            self.base_instructions = "\n".join(base_instructions_raw)
        else:
            self.base_instructions = base_instructions_raw
            
        self.conversation_starters = config_data.get("conversation_starters", [])
        self.greetings = config_data.get("greetings", {})
        self.system_messages = config_data.get("system_messages", {})
        self.rpc_messages = config_data.get("rpc_messages", {})

@lru_cache(maxsize=10)
def load_language_config(language_code: str) -> Optional[LanguageConfig]:
    """
    언어 코드에 따른 설정을 로드합니다.
    캐싱을 통해 성능을 최적화합니다.
    
    Args:
        language_code: 언어 코드 (ko, en, ja, zh)
        
    Returns:
        LanguageConfig 객체 또는 None (오류 시)
    """
    # 지원하지 않는 언어는 한국어로 fallback
    if language_code not in SUPPORTED_LANGUAGES:
        logger.warning(f"지원하지 않는 언어: {language_code}, 한국어로 fallback")
        language_code = "ko"
    
    try:
        # 현재 파일의 디렉토리를 기준으로 locales 폴더 경로 구성
        current_dir = os.path.dirname(os.path.abspath(__file__))
        locale_file = os.path.join(current_dir, "locales", f"{language_code}.json")
        
        if not os.path.exists(locale_file):
            logger.error(f"언어 파일을 찾을 수 없습니다: {locale_file}")
            # 기본 언어(한국어)로 fallback 시도
            if language_code != "ko":
                return load_language_config("ko")
            return None
        
        with open(locale_file, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        language_config = LanguageConfig(language_code, config_data)
        logger.info(f"언어 설정 로드 완료: {language_code}")
        return language_config
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON 파싱 오류 ({language_code}): {e}")
        # 기본 언어(한국어)로 fallback 시도
        if language_code != "ko":
            return load_language_config("ko")
        return None
    except Exception as e:
        logger.error(f"언어 설정 로드 실패 ({language_code}): {e}")
        # 기본 언어(한국어)로 fallback 시도
        if language_code != "ko":
            return load_language_config("ko")
        return None

def get_base_instructions(language_code: str, user_name: Optional[str] = None, context: Optional[str] = None, custom_persona: Optional[str] = None) -> str:
    """
    언어별 기본 지시사항을 반환합니다.
    
    Args:
        language_code: 언어 코드
        user_name: 사용자 이름 (선택적)
        context: 이전 대화 컨텍스트 (선택적)
        custom_persona: 커스텀 페르소나 (선택적, 빈 문자열이면 기본 페르소나 사용)
        
    Returns:
        기본 지시사항 문자열
    """
    # 커스텀 페르소나가 제공되고 비어있지 않은 경우 사용
    if custom_persona and custom_persona.strip():
        instructions = custom_persona
        logger.info(f"커스텀 페르소나 사용 (길이: {len(custom_persona)})")
    else:
        # 기본 페르소나 사용 (locales에서 로드)
        config = load_language_config(language_code)
        if not config:
            logger.error(f"언어 설정을 로드할 수 없습니다: {language_code}")
            return ""
        instructions = config.base_instructions
        logger.info(f"기본 페르소나 사용 (언어: {language_code})")
    
    # 사용자 이름 추가
    if user_name:
        instructions += f"\\n\\n사용자의 이름은 '{user_name}'입니다."
    
    # 이전 대화 컨텍스트 추가
    if context:
        instructions += f"\\n\\n이전 대화 내용:\\n{context}"
    
    return instructions

def get_conversation_starters(language_code: str) -> List[str]:
    """
    언어별 대화 시작 메시지 목록을 반환합니다.
    
    Args:
        language_code: 언어 코드
        
    Returns:
        대화 시작 메시지 리스트
    """
    config = load_language_config(language_code)
    if not config:
        logger.error(f"언어 설정을 로드할 수 없습니다: {language_code}")
        return ["Hello!"]  # 기본 메시지
    
    return config.conversation_starters

def get_greeting_message(language_code: str, user_name: Optional[str] = None, is_returning_user: bool = False) -> str:
    """
    언어별 인사 메시지를 랜덤하게 선택해서 반환합니다.
    
    Args:
        language_code: 언어 코드
        user_name: 사용자 이름 (선택적)
        is_returning_user: 재방문 사용자 여부
        
    Returns:
        랜덤하게 선택된 인사 메시지 문자열
    """
    config = load_language_config(language_code)
    if not config:
        logger.error(f"언어 설정을 로드할 수 없습니다: {language_code}")
        return "Hello!"
    
    if is_returning_user and user_name:
        # 재방문 사용자 인사
        greeting_messages = config.greetings.get("returning_user", ["Hello, {name}!"])
        if not greeting_messages:
            return "Hello!"
        
        # 랜덤하게 메시지 선택
        selected_message = random.choice(greeting_messages)
        return selected_message.format(name=user_name)
    else:
        # 새 사용자 인사
        greeting_messages = config.greetings.get("new_user", ["Hello! Please tell me your name."])
        if not greeting_messages:
            return "Hello! Please tell me your name."
        
        # 랜덤하게 메시지 선택
        selected_message = random.choice(greeting_messages)
        logger.debug(f"인사 메시지 선택됨 ({language_code}, new_user): {selected_message}")
        return selected_message

def get_system_message(language_code: str, message_key: str, **kwargs) -> str:
    """
    언어별 시스템 메시지를 반환합니다.
    
    Args:
        language_code: 언어 코드
        message_key: 메시지 키 (예: "name_saved")
        **kwargs: 메시지 포맷에 사용할 변수들
        
    Returns:
        시스템 메시지 문자열
    """
    config = load_language_config(language_code)
    if not config:
        logger.error(f"언어 설정을 로드할 수 없습니다: {language_code}")
        return "OK"
    
    message_template = config.system_messages.get(message_key, "OK")
    try:
        return message_template.format(**kwargs)
    except KeyError as e:
        logger.warning(f"메시지 포맷팅 오류: {e}, 원본 메시지 반환")
        return message_template

def clear_cache():
    """언어 설정 캐시를 클리어합니다."""
    load_language_config.cache_clear()
    logger.info("언어 설정 캐시가 클리어되었습니다.")

def get_supported_languages() -> List[str]:
    """지원하는 언어 목록을 반환합니다."""
    return SUPPORTED_LANGUAGES.copy()

def get_rpc_message(language_code: str, message_key: str) -> str:
    """
    언어별 RPC 메시지를 랜덤하게 선택해서 반환합니다.
    
    Args:
        language_code: 언어 코드
        message_key: RPC 메시지 키 (예: "check_attention", "morning_greeting")
        
    Returns:
        랜덤하게 선택된 RPC 메시지 문자열
    """
    config = load_language_config(language_code)
    if not config:
        logger.error(f"언어 설정을 로드할 수 없습니다: {language_code}")
        return "Hello!"
    
    messages = config.rpc_messages.get(message_key, [])
    if not messages:
        logger.warning(f"RPC 메시지를 찾을 수 없습니다: {message_key}")
        return "Hello!"
    
    # 랜덤하게 메시지 선택
    selected_message = random.choice(messages)
    logger.debug(f"RPC 메시지 선택됨 ({language_code}, {message_key}): {selected_message}")
    return selected_message

# 테스트 함수
def test_language_loader():
    """언어 로더 테스트 함수"""
    print("=== Language Loader Test ===")
    
    for lang in SUPPORTED_LANGUAGES:
        print(f"\\n테스트 언어: {lang}")
        
        # 기본 지시사항 테스트
        instructions = get_base_instructions(lang, "테스트유저", "이전 대화...")
        print(f"기본 지시사항 길이: {len(instructions)}")
        
        # 대화 시작 메시지 테스트
        starters = get_conversation_starters(lang)
        print(f"대화 시작 메시지 개수: {len(starters)}")
        
        # 인사 메시지 테스트
        greeting = get_greeting_message(lang, "테스트유저", True)
        print(f"재방문 인사: {greeting}")
        
        # 시스템 메시지 테스트
        sys_msg = get_system_message(lang, "name_saved", name="테스트유저")
        print(f"이름 저장 메시지: {sys_msg}")

if __name__ == "__main__":
    test_language_loader()