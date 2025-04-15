"""
Unit tests for the QuantumTokenizer module.
Tests include:
1. Token generation and verification.
2. Key rotation functionality.
3. Biometric pattern updates.
"""

import pytest
from src.tokenization.quantum_tokenizer import QuantumTokenizer

@pytest.fixture
def tokenizer():
    """Fixture to initialize a QuantumTokenizer instance."""
    return QuantumTokenizer(key_rotation_interval=10)

def test_token_generation_and_verification(tokenizer):
    """Test that tokens are generated and verified correctly."""
    data = b"test_data"
    token = tokenizer.quantum_hash(data)
    assert tokenizer.verify_token(token, data), "Token verification failed."

def test_key_rotation(tokenizer):
    """Test that keys are rotated after the specified interval."""
    import time

    data = b"test_data"
    token1 = tokenizer.quantum_hash(data)

    time.sleep(11)

    token2 = tokenizer.quantum_hash(data)

    assert not tokenizer.verify_token(token1, data), "Old token should not be valid after key rotation."

    assert tokenizer.verify_token(token2, data), "New token verification failed."

def test_biometric_pattern_update(tokenizer):
    """Test that biometric patterns are updated correctly."""
    keystroke_timings = [0.1, 0.2, 0.3]
    mouse_trajectory = [(100, 200, 0.1), (150, 250, 0.2)]

    tokenizer.update_biometric_pattern(keystroke_timings, mouse_trajectory)

    data = b"test_data"
    token = tokenizer.quantum_hash(data)

    assert tokenizer.verify_token(token, data), "Token verification failed with updated biometric pattern."

if __name__ == "__main__":
    pytest.main()
