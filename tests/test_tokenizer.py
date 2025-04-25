"""
Unit tests for the QuantumTokenizer module.
Tests include:
1. Token generation and verification.
2. Key rotation functionality.
3. Biometric pattern updates.
"""

import pytest
import time
from src.tokenization.quantum_tokenizer import QuantumTokenizer

def test_biometric_pattern_update():
    tokenizer = QuantumTokenizer()
    biometric_data = [128, 45, 92, 187]
    tokenizer.update_biometric_pattern(biometric_data)
    assert len(tokenizer.behavioral_pattern) == 32  # BLAKE2s digest size
    tokenizer.update_biometric_pattern(biometric_data, mouse_trajectory=[10, 20])
    assert len(tokenizer.behavioral_pattern) == 32


def capture_keystroke_dynamics():
    return[128, 45, 92, 187]


def test_generate_and_verify_token():
    tokenizer = QuantumTokenizer()
    biometric_data = capture_keystroke_dynamics()
    tokenizer.update_biometric_pattern(biometric_data)

    data = b"Test data for token"
    token = tokenizer.generate_token(data)
    assert isinstance(token, bytes)
    assert tokenizer.verify_token(token, data) is True

    assert tokenizer.verify_token(token, b"Different data") is False

    tokenizer.update_biometric_pattern([1, 2, 3])
    assert tokenizer.verify_token(token, data) is False


def test_key_rotation():
    tokenizer = QuantumTokenizer(key_rotation_interval=3600)  # Force immediate rotation

    biometric_data = [128, 45, 92, 187]
    tokenizer.update_biometric_pattern(biometric_data)

    data = b"Rotate test"
    token = tokenizer.generate_token(data)
    assert tokenizer.verify_token(token, data) is True

    time.sleep(2)
    tokenizer._rotate_keys_if_needed()

    assert tokenizer.verify_token(token, data) is False


if __name__ == "__main__":
    pytest.main()
