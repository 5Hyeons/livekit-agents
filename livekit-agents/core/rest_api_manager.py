"""REST API manager for wallmate-db-server integration."""

import logging
import aiohttp
from typing import Optional
from livekit.agents.llm import ChatContext, ChatMessage

logger = logging.getLogger("rest-api-manager")


class RestAPIManager:
    """
    wallmate-db-server REST API 통합 관리.

    기존 MongoDBManager + UserProfileManager를 대체하여
    모든 데이터베이스 작업을 REST API를 통해 수행.
    """

    def __init__(self, base_url: str = "http://localhost:8018"):
        """
        Initialize RestAPIManager.

        Args:
            base_url: wallmate-db-server URL
        """
        self.base_url = base_url
        logger.info(f"RestAPIManager initialized: {base_url}")

    async def _call(self, method: str, endpoint: str, json_data: dict = None) -> Optional[dict]:
        """
        Internal API 호출 헬퍼.

        Args:
            method: HTTP method (GET, POST, PATCH, DELETE)
            endpoint: API endpoint (e.g., "/api/memory/longterm/:threadId")
            json_data: Request body (optional)

        Returns:
            API response dict or None on failure
        """
        url = f"{self.base_url}{endpoint}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(method, url, json=json_data) as resp:
                    if resp.status in [200, 201]:
                        return await resp.json()
                    elif resp.status == 404:
                        logger.debug(f"Resource not found: {endpoint}")
                        return None
                    else:
                        logger.error(f"API failed: {method} {endpoint} - status {resp.status}")
                        return None
        except Exception as e:
            logger.error(f"API error: {method} {endpoint} - {e}")
            return None

    # ============================================================
    # USER PROFILE (장기 메모리)
    # ============================================================

    async def get_user_profile(self, thread_id: str, user_id: str = None, auto_create: bool = True) -> dict:
        """
        사용자 프로필 조회 (장기 메모리).
        프로필이 없으면 자동으로 생성.

        Args:
            thread_id: Thread identifier
            user_id: User identifier (auto_create=True일 때 필요)
            auto_create: 프로필 없으면 자동 생성 여부

        Returns:
            User profile dict
        """
        result = await self._call("GET", f"/api/memory/longterm/{thread_id}")

        if result and result.get("success"):
            profile = result.get("profile", {})
            logger.info(f"✅ User profile loaded: {profile.get('name', 'Unknown')}")
            return profile
        else:
            # 프로필 없으면 자동 생성
            if auto_create and user_id:
                logger.info(f"No profile found - creating new profile for {thread_id}")
                created = await self.create_user_profile(thread_id, user_id)
                if created:
                    return {"name": "Unknown", "user_id": user_id}

            logger.info("No user profile found - returning default")
            return {"name": "Unknown"}

    async def create_user_profile(self, thread_id: str, user_id: str) -> bool:
        """
        사용자 프로필 생성.

        Args:
            thread_id: Thread identifier
            user_id: User identifier

        Returns:
            Success status
        """
        result = await self._call("POST", f"/api/memory/longterm/{thread_id}", {"user_id": user_id})
        success = result and result.get("success", False)

        if success:
            logger.info(f"✅ User profile created: {thread_id}")
        else:
            logger.error(f"❌ Failed to create user profile: {thread_id}")

        return success

    async def update_user_profile(self, thread_id: str, updates: dict) -> bool:
        """
        사용자 프로필 업데이트 (특정 필드만).

        Args:
            thread_id: Thread identifier
            updates: Fields to update (e.g., {"name": "홍길동"})

        Returns:
            Success status
        """
        result = await self._call("PATCH", f"/api/memory/longterm/{thread_id}", {"updates": updates})
        success = result and result.get("success", False)

        if success:
            logger.info(f"✅ Profile updated: {list(updates.keys())}")
        else:
            logger.error(f"❌ Failed to update profile: {thread_id}")

        return success

    # ============================================================
    # CONVERSATION HISTORY (단기 메모리)
    # ============================================================

    async def get_conversation(self, thread_id: str, limit: int = 50) -> ChatContext:
        """
        대화 히스토리 조회 (단기 메모리).

        Args:
            thread_id: Thread identifier
            limit: Maximum messages to load

        Returns:
            ChatContext with loaded messages
        """
        result = await self._call("GET", f"/api/memory/conversations/{thread_id}?limit={limit}")

        chat_ctx = ChatContext()

        if result and result.get("success"):
            messages = result.get("messages", [])

            for msg in messages:
                chat_ctx.add_message(
                    role=msg["role"],
                    content=msg["content"]
                )

            logger.info(f"✅ Conversation loaded: {len(messages)} messages")
        else:
            logger.info("No conversation history - returning empty ChatContext")

        return chat_ctx

    async def save_conversation(self, thread_id: str, chat_ctx) -> bool:
        """
        대화 히스토리 저장.

        Args:
            thread_id: Thread identifier
            chat_ctx: ChatContext object (from session.history)

        Returns:
            Success status
        """
        # ChatContext.items에서 ChatMessage만 추출하여 dict로 변환
        formatted = []
        for item in chat_ctx.items:
            # ChatMessage만 저장 (FunctionCall, FunctionCallOutput 제외)
            if hasattr(item, 'role') and hasattr(item, 'content'):
                # content가 list인 경우 첫 번째 텍스트만 추출
                content = item.content
                if isinstance(content, list) and len(content) > 0:
                    content = content[0].text if hasattr(content[0], 'text') else str(content[0])

                formatted.append({
                    "role": item.role,
                    "content": content
                })

        result = await self._call("POST", f"/api/memory/conversations/{thread_id}", {"messages": formatted})
        success = result and result.get("success", False)

        if success:
            logger.info(f"✅ Conversation saved: {len(formatted)} messages")
        else:
            logger.error(f"❌ Failed to save conversation")

        return success

    # ============================================================
    # TOKEN MANAGEMENT
    # ============================================================

    async def get_token_balance(self, user_id: str) -> dict:
        """
        토큰 잔액 조회.

        Args:
            user_id: User identifier

        Returns:
            Token balance dict
        """
        result = await self._call("GET", f"/api/users/{user_id}/tokens")

        if result and result.get("success"):
            balance = {
                "current_tokens": result.get("current_tokens", 0),
                "total_earned": result.get("total_earned", 0),
                "total_spent": result.get("total_spent", 0)
            }
            logger.info(f"✅ Token balance: {balance['current_tokens']} tokens")
            return balance
        else:
            logger.warning("Failed to get token balance - returning zero")
            return {"current_tokens": 0, "total_earned": 0, "total_spent": 0}

    async def sync_token_usage(self, user_id: str, scene_id: str, amount: int, description: str) -> bool:
        """
        토큰 사용 동기화 (wallmate-db-server에 전송).

        Args:
            user_id: User identifier
            scene_id: Scene identifier
            amount: Tokens to deduct
            description: Usage description

        Returns:
            Success status
        """
        result = await self._call("DELETE", f"/api/users/{user_id}/tokens", {
            "amount": amount,
            "scene_id": scene_id,
            "description": description
        })

        if result and result.get("success"):
            new_balance = result.get("new_balance", 0)
            logger.info(f"✅ Token sync: -{amount} tokens (balance: {new_balance})")
            return True
        else:
            logger.error(f"❌ Token sync failed: {amount} tokens")
            return False


# Export
__all__ = ['RestAPIManager']