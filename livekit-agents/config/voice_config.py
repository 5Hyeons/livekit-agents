"""
Voice configuration for ElevenLabs TTS integration.
"""

from dataclasses import dataclass
from livekit.plugins import elevenlabs


@dataclass
class VoiceSettings:
    """Voice settings for ElevenLabs TTS."""
    stability: float
    similarity_boost: float
    style: float
    speed: float


@dataclass
class ElevenLabsConfig:
    """Configuration for ElevenLabs TTS with voice presets."""
    voice_id: str
    model: str
    voice_settings: VoiceSettings
    
    @classmethod
    def from_voice_name(cls, voice_name: str) -> 'ElevenLabsConfig':
        """
        Map voice name to ElevenLabs configuration.
        
        Supported voices: FEMALE_1, FEMALE_2, MALE_1, MALE_2
        
        Args:
            voice_name: Name of the voice preset
            
        Returns:
            ElevenLabsConfig instance with appropriate settings
        """
        configs = {
            "FEMALE_1": cls(
                voice_id="uyVNoMrnUku1dZyVEXwD",
                model="eleven_turbo_v2_5",
                voice_settings=VoiceSettings(
                    stability=0.5,
                    similarity_boost=0.75,
                    style=0.0,
                    speed=1.0,
                ),
            ),
            "FEMALE_2": cls(
                voice_id="uyVNoMrnUku1dZyVEXwD",
                model="eleven_turbo_v2_5",
                voice_settings=VoiceSettings(
                    stability=0.5,
                    similarity_boost=0.75,
                    style=0.0,
                    speed=1.0,
                ),
            ),
            "MALE_1": cls(
                voice_id="YBRudLRm83BV5Mazcr42",
                model="eleven_flash_v2_5",
                voice_settings=VoiceSettings(
                    stability=0.88,
                    similarity_boost=0.74,
                    style=0.34,
                    speed=1.07,
                ),
            ),
            "MALE_2": cls(
                voice_id="YBRudLRm83BV5Mazcr42",
                model="eleven_flash_v2_5",
                voice_settings=VoiceSettings(
                    stability=0.7,
                    similarity_boost=0.75,
                    style=0.0,
                    speed=1.07,
                ),
            ),
        }
        return configs.get(voice_name, configs["FEMALE_1"])
    
    def create_tts(self) -> elevenlabs.TTS:
        """
        Create ElevenLabs TTS instance with configured settings.
        
        Returns:
            Configured ElevenLabs TTS instance
        """
        return elevenlabs.TTS(
            voice_id=self.voice_id,
            model=self.model,
            voice_settings=elevenlabs.VoiceSettings(
                stability=self.voice_settings.stability,
                similarity_boost=self.voice_settings.similarity_boost,
                style=self.voice_settings.style,
                speed=self.voice_settings.speed,
            ),
            encoding="mp3_44100_32",
        )