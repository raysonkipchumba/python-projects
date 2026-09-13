import os
import sys
import bcrypt
import base64
import json
import getpass
import secrets
import string
import pyperclip
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidTag

VAULT_FILE = "vault.txt"
MASTER_HASH_FILE = "master.hash"
SALT_FILE = "salt.bin"

# Constants for encryption
ITERATIONS = 100_000
KEY_LENGTH = 32  # AES-256
NONCE_LENGTH = 12  # For AES GCM

backend = default_backend()

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def derive_key(password: bytes, salt: bytes) -> bytes:
    """Derive a symmetric encryption key from the password and salt."""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=KEY_LENGTH,
        salt=salt,
        iterations=ITERATIONS,
        backend=backend
    )
    return kdf.derive(password)

def encrypt(plaintext: bytes, key: bytes) -> bytes:
    """Encrypt plaintext using AES-GCM."""
    nonce = secrets.token_bytes(NONCE_LENGTH)
    cipher = Cipher(algorithms.AES(key), modes.GCM(nonce), backend=backend)
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(plaintext) + encryptor.finalize()
    return nonce + encryptor.tag + ciphertext

def decrypt(ciphertext: bytes, key: bytes) -> bytes:
    """Decrypt ciphertext using AES-GCM."""
    nonce = ciphertext[:NONCE_LENGTH]
    tag = ciphertext[NONCE_LENGTH:NONCE_LENGTH+16]
    actual_ciphertext = ciphertext[NONCE_LENGTH+16:]
    cipher = Cipher(algorithms.AES(key), modes.GCM(nonce, tag), backend=backend)
    decryptor = cipher.decryptor()
    try:
        plaintext = decryptor.update(actual_ciphertext) + decryptor.finalize()
    except InvalidTag:
        raise ValueError("Invalid decryption key or corrupted data.")
    return plaintext

def save_master_password(password: str):
    """Hash and save the master password with bcrypt."""
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    with open(MASTER_HASH_FILE, "wb") as f:
        f.write(hashed)

def verify_master_password(password: str) -> bool:
    """Verify the master password against the stored bcrypt hash."""
    if not os.path.exists(MASTER_HASH_FILE):
        return False
    with open(MASTER_HASH_FILE, "rb") as f:
        hashed = f.read()
    return bcrypt.checkpw(password.encode(), hashed)

def load_salt() -> bytes:
    """Load or generate salt for key derivation."""
    if os.path.exists(SALT_FILE):
        with open(SALT_FILE, "rb") as f:
            return f.read()
    else:
        salt = secrets.token_bytes(16)
        with open(SALT_FILE, "wb") as f:
            f.write(salt)
        return salt

def load_vault(key: bytes) -> dict:
    """Load and decrypt the vault data."""
    if not os.path.exists(VAULT_FILE):
        return {}
    with open(VAULT_FILE, "rb") as f:
        encrypted_data = f.read()
    if not encrypted_data:
        return {}
    decrypted = decrypt(encrypted_data, key)
    return json.loads(decrypted.decode())

def save_vault(data: dict, key: bytes):
    """Encrypt and save the vault data."""
    plaintext = json.dumps(data).encode()
    encrypted = encrypt(plaintext, key)
    with open(VAULT_FILE, "wb") as f:
        f.write(encrypted)

def generate_password(length=16, use_upper=True, use_lower=True, use_digits=True, use_symbols=True) -> str:
    """Generate a strong random password."""
    char_sets = []
    if use_upper:
        char_sets.append(string.ascii_uppercase)
    if use_lower:
        char_sets.append(string.ascii_lowercase)
    if use_digits:
        char_sets.append(string.digits)
    if use_symbols:
        char_sets.append("!@#$%^&*()-_=+[]{}|;:,.<>?")

    if not char_sets:
        raise ValueError("At least one character set must be selected")

    all_chars = "".join(char_sets)
    password = "".join(secrets.choice(all_chars) for _ in range(length))
    return password

def prompt_master_password(new=False) -> str:
    """Prompt user for master password."""
    while True:
        pw = getpass.getpass("Enter master password: ")
        if new:
            pw_confirm = getpass.getpass("Confirm master password: ")
            if pw != pw_confirm:
                print("Passwords do not match. Try again.")
                continue
        return pw

def print_help():
    print("""
Available commands:
  help                   Show this help menu
  add                    Add a new account
  edit <account>         Edit an existing account
  delete <account>       Delete an account
  list                   List all accounts
  search <term>          Search accounts by name or username
  generate               Generate a strong password
  copy <account>         Copy password of an account to clipboard
  changepw               Change master password
  lock                   Lock the vault
  logout                 Logout and exit
  exit                   Exit without logout
""")

def main():
    clear_screen()
    print("Welcome to CLI Password Manager")
    if not os.path.exists(MASTER_HASH_FILE):
        print("No master password set. Please create one.")
        master_password = prompt_master_password(new=True)
        save_master_password(master_password)
        print("Master password set successfully.")
    else:
        for _ in range(3):
            master_password = prompt_master_password()
            if verify_master_password(master_password):
                break
            else:
                print("Incorrect password.")
        else:
            print("Too many failed attempts. Exiting.")
            sys.exit(1)

    salt = load_salt()
    key = derive_key(master_password.encode(), salt)

    vault = load_vault(key)

    locked = False

    while True:
        if locked:
            cmd = input("[LOCKED] Enter 'unlock' to unlock: ").strip()
            if cmd == "unlock":
                pw = getpass.getpass("Enter master password to unlock: ")
                if verify_master_password(pw):
                    key = derive_key(pw.encode(), salt)
                    vault = load_vault(key)
                    locked = False
                    print("Vault unlocked.")
                else:
                    print("Incorrect password.")
            elif cmd == "exit":
                print("Exiting.")
                break
            else:
                print("Vault is locked. Use 'unlock' or 'exit'.")
            continue

        try:
            cmd_line = input("manager> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not cmd_line:
            continue

        parts = cmd_line.split()
        cmd = parts[0].lower()
        args = parts[1:]

        if cmd == "help":
            print_help()

        elif cmd == "add":
            account = input("Account name: ").strip()
            if not account:
                print("Account name cannot be empty.")
                continue
            if account in vault:
                print("Account already exists.")
                continue
            username = input("Username: ").strip()
            password = getpass.getpass("Password (leave empty to generate): ")
            if not password:
                password = generate_password()
                print(f"Generated password: {password}")
            url = input("URL (optional): ").strip()
            notes = input("Notes (optional): ").strip()
            vault[account] = {
                "username": username,
                "password": password,
                "url": url,
                "notes": notes,
            }
            save_vault(vault, key)
            print(f"Account '{account}' added.")

        elif cmd == "edit":
            if not args:
                print("Usage: edit <account>")
                continue
            account = args[0]
            if account not in vault:
                print("Account not found.")
                continue
            print(f"Editing account '{account}'. Press enter to keep current value.")
            username = input(f"Username [{vault[account]['username']}]: ").strip()
            if username:
                vault[account]['username'] = username
            password = getpass.getpass("Password (leave empty to keep current): ")
            if password:
                vault[account]['password'] = password
            url = input(f"URL [{vault[account]['url']}]: ").strip()
            if url:
                vault[account]['url'] = url
            notes = input(f"Notes [{vault[account]['notes']}]: ").strip()
            if notes:
                vault[account]['notes'] = notes
            save_vault(vault, key)
            print(f"Account '{account}' updated.")

        elif cmd == "delete":
            if not args:
                print("Usage: delete <account>")
                continue
            account = args[0]
            if account not in vault:
                print("Account not found.")
                continue
            confirm = input(f"Are you sure you want to delete '{account}'? (yes/no): ").strip().lower()
            if confirm == "yes":
                del vault[account]
                save_vault(vault, key)
                print(f"Account '{account}' deleted.")
            else:
                print("Delete cancelled.")

        elif cmd == "list":
            if not vault:
                print("No accounts stored.")
                continue
            print(f"{'Account':20} {'Username':20} {'URL':30}")
            print("-" * 70)
            for acc, data in vault.items():
                print(f"{acc:20} {data['username']:20} {data['url'][:30]:30}")

        elif cmd == "search":
            if not args:
                print("Usage: search <term>")
                continue
            term = args[0].lower()
            results = []
            for acc, data in vault.items():
                if term in acc.lower() or term in data['username'].lower():
                    results.append((acc, data))
            if not results:
                print("No matching accounts found.")
                continue
            print(f"{'Account':20} {'Username':20} {'URL':30}")
            print("-" * 70)
            for acc, data in results:
                print(f"{acc:20} {data['username']:20} {data['url'][:30]:30}")

        elif cmd == "generate":
            try:
                length = int(input("Password length (default 16): ") or "16")
                use_upper = input("Use uppercase? (Y/n): ").strip().lower() != "n"
                use_lower = input("Use lowercase? (Y/n): ").strip().lower() != "n"
                use_digits = input("Use digits? (Y/n): ").strip().lower() != "n"
                use_symbols = input("Use symbols? (Y/n): ").strip().lower() != "n"
                pwd = generate_password(length, use_upper, use_lower, use_digits, use_symbols)
                print(f"Generated password: {pwd}")
                copy = input("Copy to clipboard? (Y/n): ").strip().lower() != "n"
                if copy:
                    pyperclip.copy(pwd)
                    print("Password copied to clipboard.")
            except Exception as e:
                print(f"Error generating password: {e}")

        elif cmd == "copy":
            if not args:
                print("Usage: copy <account>")
                continue
            account = args[0]
            if account not in vault:
                print("Account not found.")
                continue
            pyperclip.copy(vault[account]['password'])
            print(f"Password for '{account}' copied to clipboard.")

        elif cmd == "changepw":
            old_pw = getpass.getpass("Enter current master password: ")
            if not verify_master_password(old_pw):
                print("Incorrect current password.")
                continue
            new_pw = prompt_master_password(new=True)
            # Re-encrypt vault with new key
            new_salt = secrets.token_bytes(16)
            with open(SALT_FILE, "wb") as f:
                f.write(new_salt)
            new_key = derive_key(new_pw.encode(), new_salt)
            save_vault(vault, new_key)
            save_master_password(new_pw)
            salt = new_salt
            key = new_key
            print("Master password changed successfully.")

        elif cmd == "lock":
            locked = True
            vault = {}
            print("Vault locked.")

        elif cmd == "logout":
            print("Logging out and exiting.")
            break

        elif cmd == "exit":
            print("Exiting without logout.")
            break

        else:
            print("Unknown command. Type 'help' for a list of commands.")

if __name__ == "__main__":
    main()
