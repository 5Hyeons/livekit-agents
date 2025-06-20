# AnimationData Format Specification

## Overview
AnimationData는 얼굴 애니메이션 블렌드쉐입 데이터를 전송하는 형식입니다. 
기본적으로 블렌드쉐입 데이터만 포함하거나, 오디오와 함께 페어로 전송할 수 있습니다.

## Data Types

### Type 0: Animation Only (TYPE_ANIMATION_ONLY)
기존 방식과 동일하게 블렌드쉐입 데이터만 포함합니다.
- 데이터: float32 배열의 바이트 표현 (52개 특성)

### Type 1: Animation + Audio Pair (TYPE_ANIMATION_AUDIO_PAIR)
애니메이션과 오디오 데이터를 하나의 패킷으로 전송합니다.

#### Header Structure (8 bytes)
| Offset | Size | Type  | Description |
|--------|------|-------|-------------|
| 0      | 4    | uint32 | 애니메이션 데이터 길이 (bytes) |
| 4      | 4    | uint32 | 오디오 데이터 길이 (bytes) |

> **Why so small?**  샘플레이트·채널수 같은 메타데이터는 바이트 스트림의 attributes( `lk.animation_sample_rate`, `lk.animation_segment_id` 등) 로 전송합니다.  
> 헤더는 실제 payload 크기만 알려주어 파싱이 단순-고속화됩니다.

#### Data Section
1. Header 뒤 ‑ 애니메이션 데이터 (float32, `num_features` × 4 bytes)
2. 이어서 ‑ 오디오 데이터 (int16 PCM, 원본 샘플레이트 그대로)

## Client Implementation Example (Python)

```python
import struct, numpy as np
from livekit.agents.types import ATTRIBUTE_ANIMATION_SAMPLE_RATE

def parse_animation_pair(data: bytes, attrs: dict[str, str]):
    anim_len, audio_len = struct.unpack_from('<II', data, 0)
    anim_start = 8
    anim_end   = anim_start + anim_len
    audio_start = anim_end
    audio_end   = audio_start + audio_len

    # blendshapes
    blend = np.frombuffer(data[anim_start:anim_end], dtype=np.float32)
    # audio (int16)
    audio = np.frombuffer(data[audio_start:audio_end], dtype=np.int16)

    sample_rate = int(attrs.get(ATTRIBUTE_ANIMATION_SAMPLE_RATE, 48000))
    return blend, audio, sample_rate
```

## Unity/C# Implementation Example

```csharp
public static void ParseAnimationPair(byte[] data, int sampleRate,
    out float[] blend, out short[] audio)
{
    uint animLen = BitConverter.ToUInt32(data, 0);
    uint audioLen = BitConverter.ToUInt32(data, 4);

    int animStart = 8;
    int audioStart = animStart + (int)animLen;

    blend = new float[animLen / 4];
    Buffer.BlockCopy(data, animStart, blend, 0, (int)animLen);

    audio = new short[audioLen / 2];
    Buffer.BlockCopy(data, audioStart, audio, 0, (int)audioLen);
}
```

## Notes
- 모든 multi-byte 값은 little-endian 형식입니다.
- 오디오 데이터는 원본 샘플레이트 그대로 전송됩니다 (리샘플링 없음).
- 블렌드쉐입 데이터는 항상 52개의 float32 값입니다.
- 오디오는 int16 PCM 형식입니다. 