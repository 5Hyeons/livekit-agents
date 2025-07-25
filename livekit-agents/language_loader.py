"""
Deprecated language loader module.

This module is kept for backward compatibility but is no longer used.
The new persona-based system uses dynamic response generation instead of fixed messages.
"""

import logging
from typing import List

logger = logging.getLogger("language-loader")

# 지원하는 언어 목록
SUPPORTED_LANGUAGES = ["ko", "en", "ja", "zh"]


def get_supported_languages() -> List[str]:
    """
    지원하는 언어 목록을 반환합니다.
    
    Returns:
        지원하는 언어 코드 리스트
    """
    return SUPPORTED_LANGUAGES.copy()


# Note: This module is deprecated. Use the new persona-based system instead.
# The following functions have been removed:
# - get_base_instructions: Use config.base_instructions.create_base_instructions
# - get_conversation_starters: Use dynamic generate_reply with system context
# - get_greeting_message: Use dynamic generate_reply with system context  
# - get_system_message: Use dynamic generate_reply with system context
# - get_rpc_message: Use dynamic generate_reply with system context