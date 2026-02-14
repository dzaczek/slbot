"""
Unit tests for ws_protocol.py — binary packet encoding/decoding.

Run: python -m pytest test_ws_protocol.py -v
  or: python test_ws_protocol.py
"""

import math
import struct
import pytest

from ws_protocol import (
    PacketReader,
    decode_angle, decode_angle_16, decode_speed, decode_fam, decode_relative_pos,
    encode_angle, encode_angle_precise, encode_login,
    encode_boost_start, encode_boost_stop,
    TWO_PI,
)


# ─── PacketReader tests ─────────────────────────────────────────────────

class TestPacketReader:
    def test_read_uint8(self):
        r = PacketReader(bytes([0, 127, 255]))
        assert r.read_uint8() == 0
        assert r.read_uint8() == 127
        assert r.read_uint8() == 255

    def test_read_int8(self):
        r = PacketReader(bytes([0, 127, 128, 255]))
        assert r.read_int8() == 0
        assert r.read_int8() == 127
        assert r.read_int8() == -128
        assert r.read_int8() == -1

    def test_read_uint16(self):
        r = PacketReader(struct.pack('>HH', 0, 65535))
        assert r.read_uint16() == 0
        assert r.read_uint16() == 65535

    def test_read_int16(self):
        r = PacketReader(struct.pack('>hh', -1000, 1000))
        assert r.read_int16() == -1000
        assert r.read_int16() == 1000

    def test_read_uint24(self):
        # 0x0186A0 = 100000
        r = PacketReader(bytes([0x01, 0x86, 0xA0]))
        assert r.read_uint24() == 100000

    def test_read_int24_positive(self):
        r = PacketReader(bytes([0x01, 0x86, 0xA0]))
        assert r.read_int24() == 100000

    def test_read_int24_negative(self):
        # -1 in 24-bit signed = 0xFFFFFF
        r = PacketReader(bytes([0xFF, 0xFF, 0xFF]))
        assert r.read_int24() == -1

    def test_read_uint32(self):
        r = PacketReader(struct.pack('>I', 123456789))
        assert r.read_uint32() == 123456789

    def test_remaining(self):
        r = PacketReader(bytes([1, 2, 3, 4, 5]))
        assert r.remaining == 5
        r.read_uint8()
        assert r.remaining == 4
        r.skip(2)
        assert r.remaining == 2

    def test_read_string_fixed(self):
        r = PacketReader(b'Hello\x00World')
        assert r.read_string(5) == 'Hello'

    def test_read_string_null_terminated(self):
        r = PacketReader(b'Hello\x00World')
        assert r.read_string() == 'Hello'


# ─── Decode tests ────────────────────────────────────────────────────────

class TestDecode:
    def test_decode_angle_zero(self):
        assert decode_angle(0) == 0.0

    def test_decode_angle_quarter(self):
        # 64 = 256/4 → pi/2
        result = decode_angle(64)
        assert abs(result - math.pi / 2) < 0.03

    def test_decode_angle_half(self):
        # 128 = 256/2 → pi
        result = decode_angle(128)
        assert abs(result - math.pi) < 0.03

    def test_decode_angle_three_quarters(self):
        # 192 = 256*3/4 → 3*pi/2
        result = decode_angle(192)
        assert abs(result - 3 * math.pi / 2) < 0.03

    def test_decode_angle_full_circle(self):
        # 255 is close to 2*pi but not exactly
        result = decode_angle(255)
        assert result < TWO_PI
        assert result > 6.0

    def test_decode_angle_16_zero(self):
        assert decode_angle_16(0) == 0.0

    def test_decode_angle_16_half(self):
        result = decode_angle_16(32768)
        assert abs(result - math.pi) < 0.001

    def test_decode_speed(self):
        assert decode_speed(0) == 0.0
        assert abs(decode_speed(18) - 1.0) < 0.001
        assert abs(decode_speed(180) - 10.0) < 0.001

    def test_decode_fam(self):
        assert decode_fam(0) == 0.0
        assert abs(decode_fam(16777215) - 1.0) < 0.001
        result = decode_fam(8388607)
        assert 0.49 < result < 0.51

    def test_decode_relative_pos_center(self):
        # 128 → 0.0
        assert decode_relative_pos(128) == 0.0

    def test_decode_relative_pos_negative(self):
        # 0 → (0 - 128) / 2 = -64.0
        assert decode_relative_pos(0) == -64.0

    def test_decode_relative_pos_positive(self):
        # 255 → (255 - 128) / 2 = 63.5
        assert decode_relative_pos(255) == 63.5


# ─── Encode tests ────────────────────────────────────────────────────────

class TestEncode:
    def test_encode_angle_zero(self):
        data = encode_angle(0.0)
        assert data[0] == 251  # Packet type
        assert data[1] == 0    # Angle byte

    def test_encode_angle_pi(self):
        data = encode_angle(math.pi)
        assert data[0] == 251
        assert data[1] == 128  # pi → 128

    def test_encode_angle_half_pi(self):
        data = encode_angle(math.pi / 2)
        assert data[0] == 251
        assert data[1] == 64  # pi/2 → 64

    def test_encode_angle_negative_wraps(self):
        # -pi/2 should wrap to 3*pi/2 → 192
        data = encode_angle(-math.pi / 2)
        assert data[0] == 251
        assert data[1] == 192

    def test_encode_angle_precise(self):
        data = encode_angle_precise(math.pi)
        assert data[0] == 252  # Packet type
        val = struct.unpack('>H', data[1:3])[0]
        assert val == 32768  # pi → 32768

    def test_encode_decode_angle_roundtrip(self):
        """Encode then decode should be close to original (within quantization)."""
        for angle in [0.0, math.pi / 4, math.pi / 2, math.pi, 3 * math.pi / 2]:
            encoded = encode_angle(angle)
            decoded = decode_angle(encoded[1])
            # uint8 quantization error is up to TWO_PI/256 ≈ 0.0245
            assert abs(decoded - (angle % TWO_PI)) < 0.03, f"Roundtrip failed for {angle}"

    def test_encode_login(self):
        data = encode_login("TestBot", skin_id=5)
        assert data[0] == 115   # Login packet type
        assert data[1] == 11    # Protocol version
        assert data[2] == 5     # Skin ID
        assert data[3:].decode('utf-8') == "TestBot"

    def test_encode_login_truncates_long_name(self):
        long_name = "A" * 50
        data = encode_login(long_name)
        # Name should be truncated to 24 bytes
        assert len(data) == 3 + 24

    def test_encode_boost_start(self):
        assert encode_boost_start() == bytes([253])

    def test_encode_boost_stop(self):
        assert encode_boost_stop() == bytes([254])


# ─── Edge cases ──────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_packet_reader(self):
        r = PacketReader(b'')
        assert r.remaining == 0

    def test_reader_offset(self):
        r = PacketReader(bytes([10, 20, 30, 40, 50]), offset=2)
        assert r.read_uint8() == 30
        assert r.remaining == 2

    def test_decode_angle_all_values(self):
        """All 256 possible angle bytes should decode to [0, 2*pi)."""
        for i in range(256):
            angle = decode_angle(i)
            assert 0.0 <= angle < TWO_PI, f"decode_angle({i}) = {angle} out of range"

    def test_decode_speed_range(self):
        """Speed should be non-negative for all byte values."""
        for i in range(256):
            speed = decode_speed(i)
            assert speed >= 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
