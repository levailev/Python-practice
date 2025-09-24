import hashlib

def hash_text(text: str, algorithm: str = "sha256") -> str:
 
    # Create a hash object from hashlib
    try:
        hash_obj = hashlib.new(algorithm)
    except ValueError:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    # Encode text to bytes and update the hash object
    hash_obj.update(text.encode('utf-8'))
    return hash_obj.hexdigest()

# --- Example usage ---
if __name__ == "__main__":
    text = "secretPassword_!12"
    
    print("SHA1   :", hash_text(text, "sha1"))
    print("SHA224 :", hash_text(text, "sha224"))
    print("SHA256 :", hash_text(text, "sha256"))
    print("SHA384 :", hash_text(text, "sha384"))
    print("SHA512 :", hash_text(text, "sha512"))