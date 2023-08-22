import typing
import itertools
import hashlib
import math


INTEGRITY_HASH_SIZE = 64  # 64 hex digits is 256 bits, so half of sha512 hexdigest.

Hash = typing.Any


def _integrity_hash_internals(
        b: bytes, 
        hash: typing.Optional[str] = None
        ) -> tuple[str, Hash]:
    m = hashlib.sha512()
    m.update(b)
    actual_hash = m.hexdigest()[:INTEGRITY_HASH_SIZE]
    if hash is not None:
        assert actual_hash == hash, (
            "Provided bytes input incompatible with `hash`")
    return actual_hash, m


def get_integrity_hash(b: bytes) -> str:
    """Get the hash that should be passed alongside `b` to assert its integrity"""
    hash, _ = _integrity_hash_internals(b)
    return hash


def assert_integrity(b: bytes, hash: str) -> None:
    """Ensure that `b` is *exactly* what it is expected to be"""
    _integrity_hash_internals(b, hash)


def not_bad_pad(
        bad_pad: bytes, 
        pad_length: int,
        hash: typing.Optional[str] = None
        ) -> bytes:
    """Converts `bad_pad` into a "not bad" substitute to a one-time-pad

    One-time pad (OTP) is the name of both 1. a kind of pre-shared keys and of
    2. a family of encryption techniques making use of such pre-shared keys.
    For bytes `plaintext` and `pad` with `len(plaintext) == len(pad)`, the
    encryption and decryption are as simple as `ciphertext = plaintext XOR pad`
    and `plaintext = ciphertext XOR pad`.

    Proper OTP encryption cannot be cracked, but imposes rather harsh
    constraints on the `pad` key, including:

    *   the `pad` must be at least as long as the `plaintext`;
    *   the `pad` must be uniformly distributed;
    *   the `pad` must be independent from the `plaintext`; and
    *   the `pad` must be high entropy, free of patterns.

    Suppose that we have a `bad_pad` that, if we were to use it directly as the
    `pad`, could potentially fail all the above requirements. The present
    functions aims to **convert such a `bad_pad` to a "not bad" `pad`**. This
    "not bad" version still does not perfectly satisfying these requirements
    (so we lose the "cannot be cracked" property), but it is definitely a step
    in the right direction.

    In doing so, we impose further requirements:

    *   any small change in `bad_pad` should, with high probability, have a
        large effect on the resulting `pad`;
    *   one should be able to assess (with high probability) if they've got
        the "right" `bad_pad` required to obtain the right `pad`; and
    *   even if `plaintext` has low entropy and/or is very non-independent
        from `bad_pad', access to `ciphertext = plaintext XOR pad` should not
        reveal much information about `bad_pad`.

    This implementation uses `hashlib.sha512` to address all these concerns.

    *   We first process the whole `bad_pad` and use half of its digest to
        assess integrity, thus keeping 256 "secret" bits in the hashing
        algorithm's state. Even with the integrity hash in hands, the
        generated pad will still be highly sensitive to any perturbation
        anywhere in `bad_pad`.
    *   We then iteratively feed "chunks" of the `bad_pad` to the hashing
        algorithm, re-using the `bad_pad` as often as required.
    *   The first half of the digest after each chunk is concatenated to
        to obtain the desired pad. In the event of an "all-zero" `plaintext`,
        thus exposing the `pad == ciphertext`, it would still be hard to
        extract any meaningful information from `bad_pad`.
    """
    EATEN_BYTES_PER_CHUNK = 131  # Smallest prime larger than 128
    YIELDED_BYTES_PER_CHUNK = 32  # Half of the 64-bytes digest of sha512
    chunks_needed = math.ceil(pad_length / YIELDED_BYTES_PER_CHUNK)

    # Check integrity hash and initialize the algorithm's state
    _, m = _integrity_hash_internals(bad_pad, hash)
    
    # Iterator looping over `bad_pad` forever
    def infinite_bad_pad():
        assert len(bad_pad) > 0
        while True:
            for element in bad_pad:
                yield element

    # Iterator processing chunks of `bad_pad` to produce the `pad``
    def chunks():
        it = infinite_bad_pad()
        for _ in itertools.repeat(None, chunks_needed):
            m.update(bytes(itertools.islice(it, EATEN_BYTES_PER_CHUNK)))
            yield m.digest()[:YIELDED_BYTES_PER_CHUNK]
    
    # Do the stitching and drop any extra
    pad = b"".join(chunks())
    return pad[:pad_length]


def bytes_xor(a: bytes, b: bytes) -> bytes:
    assert len(a) == len(b)
    return (
        int.from_bytes(a, 'little') ^ int.from_bytes(b, 'little')
        ).to_bytes(len(a), 'little')


def encrypt_or_decrypt(
        input: bytes,
        key: bytes, 
        *,
        input_hash: typing.Optional[str] = None,
        key_hash: typing.Optional[str] = None,
        output_hash: typing.Optional[str] = None
        ) -> bytes:
    """Encrypt (or decrypt) `input` using `key`
    
    This function can perform both encryption and decryption.

    Integrity checks are performed for the `input` (plaintext or ciphertext),
    `key` (from which we'll derive an Ersatz to a one-time pad) and/or output 
    (plaintext or ciphertext) whenever a corresponding hash is provided.
    """
    if input_hash is not None:
        assert_integrity(input, input_hash)

    pad = not_bad_pad(key, len(input), key_hash)
    output = bytes_xor(input, pad)

    if output_hash is not None:
        assert_integrity(output, output_hash)
    
    return output


