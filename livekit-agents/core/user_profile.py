"""MongoDB-based user profile management with credits balance integration."""

import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional
import aiohttp
import asyncio

# MongoDB store will be passed as parameter

logger = logging.getLogger("user-profile")


class UserProfileManager:
    """MongoDB Store 기반 사용자 프로필 관리."""
    
    @staticmethod
    async def get_profile(user_identity: str, thread_identity: str, store) -> Dict[str, Any]:
        """
        Get user profile with credits balance from MongoDB store and wallmate-db-server.
        Creates thread profile if it doesn't exist.

        Args:
            user_identity: The user's identity
            thread_identity: The thread's identity (e.g., "scene_user-123")
            store: MongoDB store instance

        Returns:
            User profile data with integrated credits info

        Raises:
            ValueError: If user_identity is missing or credits balance fails
        """
        if not user_identity:
            raise ValueError("user_identity is required")

        try:
            # 1. Get profile from MongoDB Store
            profile_data = store.get(
                namespace=(thread_identity,),
                key="basic_info"
            )

            if not profile_data or not profile_data.value:
                # Profile doesn't exist - create it via wallmate-db-server
                logger.info(f"[UserProfile] Thread profile not found for: {thread_identity}, creating new one")

                created = await UserProfileManager.create_thread_profile(thread_identity, user_identity)
                if not created:
                    raise ValueError(f"Failed to create thread profile for: {thread_identity}")

                # Try to get profile again after creation
                profile_data = store.get(
                    namespace=(thread_identity,),
                    key="basic_info"
                )

                if not profile_data or not profile_data.value:
                    raise ValueError(f"Thread profile still not found after creation: {thread_identity}")

            profile = profile_data.value
            logger.info(f"[UserProfile] Loaded profile for thread: {thread_identity} (user: {user_identity})")

            # 2. Get credits balance from wallmate-db-server
            credit_info = await UserProfileManager.get_credits_balance(user_identity)

            if not credit_info:
                raise ValueError(f"Failed to get credits balance for user: {user_identity}")

            # 3. Integrate credits info into profile
            profile["credit_info"] = credit_info
            profile["thread_id"] = thread_identity
            profile["user_id"] = user_identity

            logger.info(f"[UserProfile] Complete profile loaded with {credit_info['remaining']} credits for: {user_identity}")
            return profile

        except ValueError:
            # Re-raise ValueError as is
            raise
        except Exception as e:
            logger.error(f"[UserProfile] Unexpected error getting profile for {user_identity}: {e}")
            raise ValueError(f"Failed to get profile for {user_identity}: {str(e)}")
    
    @staticmethod
    def update_user_name(thread_identity: str, name: str, store) -> bool:
        """
        Update user's name in their profile.
        
        Args:
            thread_identity: The thread's identity
            name: The new name to save
            store: MongoDB store instance
        
        Returns:
            True if successful, False otherwise
        """
        if not thread_identity:
            return False
        
        try:
            
            # Get existing profile
            profile_data = store.get(
                namespace=(thread_identity,),
                key="basic_info"
            )
            
            if profile_data and profile_data.value:
                # Update existing profile
                updated_profile = profile_data.value
                old_name = updated_profile.get("name", "Unknown")
                updated_profile["name"] = name
                updated_profile["last_seen"] = datetime.now().isoformat()
                
                # Save updated profile to MongoDB Store
                store.put(
                    namespace=(thread_identity,),
                    key="basic_info",
                    value=updated_profile
                )
                
                logger.info(f"[UserProfile] Updated user name: {thread_identity} ({old_name} -> {name})")
                return True
            else:
                logger.warning(f"[UserProfile] Profile not found for update: {thread_identity}")
                return False
                
        except Exception as e:
            logger.error(f"[UserProfile] Failed to update user name for {thread_identity}: {e}")
            return False

    @staticmethod
    async def get_credits_balance(user_identity: str) -> Optional[Dict[str, Any]]:
        """
        Get credits balance from wallmate-db-server for session startup.

        Args:
            user_identity: The user's identity (user_id)

        Returns:
            Credits balance data or None if failed
        """
        base_url = os.getenv("WALLMATE_DB_SERVER_URL", "http://localhost:8028")

        try:
            timeout = aiohttp.ClientTimeout(total=5.0)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                url = f"{base_url}/api/credits/balance/{user_identity}"

                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("success"):
                            logger.info(f"[UserProfile] Retrieved credits balance for {user_identity}: {data['current_credits']} credits")
                            return {
                                "remaining": data["current_credits"],
                                "total_earned": data["total_earned"],
                                "total_spent": data["total_spent"],
                                "status": "normal",
                                "credit_to_deduct": 0
                            }
                        else:
                            logger.error(f"[UserProfile] Credits balance API returned failure: {data}")
                    else:
                        logger.error(f"[UserProfile] Credits balance API returned status {response.status}")

        except asyncio.TimeoutError:
            logger.error(f"[UserProfile] Timeout getting credits balance for {user_identity}")
        except Exception as e:
            logger.error(f"[UserProfile] Error getting credits balance for {user_identity}: {e}")

        return None

    @staticmethod
    async def create_thread_profile(thread_id: str, user_id: str) -> bool:
        """
        Create thread profile via wallmate-db-server API.

        Args:
            thread_id: Thread identifier (e.g., "scene_user-123")
            user_id: User identifier

        Returns:
            True if successful, False otherwise
        """
        base_url = os.getenv("WALLMATE_DB_SERVER_URL", "http://localhost:8028")

        try:
            timeout = aiohttp.ClientTimeout(total=5.0)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                url = f"{base_url}/api/memory/thread/profile"

                payload = {
                    "thread_id": thread_id,
                    "user_id": user_id
                }

                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("success"):
                            logger.info(f"[UserProfile] Created thread profile for: {thread_id} (user: {user_id})")
                            return True
                        else:
                            logger.error(f"[UserProfile] Thread profile creation API returned failure: {data}")
                    else:
                        logger.error(f"[UserProfile] Thread profile creation API returned status {response.status}")

        except asyncio.TimeoutError:
            logger.error(f"[UserProfile] Timeout creating thread profile for: {thread_id}")
        except Exception as e:
            logger.error(f"[UserProfile] Error creating thread profile for {thread_id}: {e}")

        return False

    @staticmethod
    async def sync_credits_usage(user_identity: str, scene_id: str, credits_used: int, description: str = "Session usage") -> bool:
        """
        Sync credits usage to wallmate-db-server at session end.

        Args:
            user_identity: The user's identity (user_id)
            scene_id: Scene identifier for tracking scene-specific usage
            credits_used: Total credits used during session
            description: Description of usage

        Returns:
            True if successful, False otherwise
        """
        if credits_used <= 0:
            logger.debug(f"[UserProfile] No credits used for {user_identity}, skipping sync")
            return True

        base_url = os.getenv("WALLMATE_DB_SERVER_URL", "http://localhost:8028")

        try:
            timeout = aiohttp.ClientTimeout(total=10.0)  # Longer timeout for final sync
            async with aiohttp.ClientSession(timeout=timeout) as session:
                url = f"{base_url}/api/credits/use"

                payload = {
                    "user_id": user_identity,
                    "scene_id": scene_id,
                    "amount": credits_used,
                    "description": description,
                    "item_name": "Voice Session"
                }

                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("success"):
                            logger.info(f"[UserProfile] Synced credits usage for {user_identity}: {credits_used} credits used")
                            return True
                        else:
                            logger.error(f"[UserProfile] Credits sync API returned failure: {data}")
                    else:
                        logger.error(f"[UserProfile] Credits sync API returned status {response.status}")

        except asyncio.TimeoutError:
            logger.error(f"[UserProfile] Timeout syncing credits usage for {user_identity}")
        except Exception as e:
            logger.error(f"[UserProfile] Error syncing credits usage for {user_identity}: {e}")

        return False

