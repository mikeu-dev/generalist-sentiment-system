"""
Security utilities untuk validasi dan sanitasi input
"""
import os
import hashlib
from datetime import datetime
from werkzeug.utils import secure_filename
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Konfigurasi keamanan
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB
MAX_FILENAME_LENGTH = 255


def allowed_file(filename: str) -> bool:
    """
    Validasi apakah file extension diizinkan.
    
    Args:
        filename: Nama file yang akan divalidasi
        
    Returns:
        True jika extension valid, False jika tidak
    """
    if not filename or '.' not in filename:
        return False
    
    extension = filename.rsplit('.', 1)[1].lower()
    return extension in ALLOWED_EXTENSIONS


def validate_and_secure_filename(filename: str, max_length: int = MAX_FILENAME_LENGTH) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Validasi dan amankan filename.
    
    Args:
        filename: Nama file original
        max_length: Panjang maksimum filename
        
    Returns:
        Tuple (is_valid, secured_filename, error_message)
    """
    if not filename:
        return False, None, "Filename kosong"
    
    # Validasi extension
    if not allowed_file(filename):
        allowed = ', '.join(ALLOWED_EXTENSIONS)
        return False, None, f"Format file tidak diizinkan. Gunakan: {allowed}"
    
    # Secure filename (hapus karakter berbahaya)
    safe_filename = secure_filename(filename)
    
    if not safe_filename:
        return False, None, "Filename tidak valid setelah sanitasi"
    
    # Validasi panjang
    if len(safe_filename) > max_length:
        return False, None, f"Nama file terlalu panjang (max {max_length} karakter)"
    
    # Tambahkan timestamp untuk menghindari collision
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name, ext = safe_filename.rsplit('.', 1)
    unique_filename = f"{timestamp}_{name}.{ext}"
    
    return True, unique_filename, None


def validate_file_size(file_stream, max_size: int = MAX_FILE_SIZE) -> Tuple[bool, Optional[str]]:
    """
    Validasi ukuran file.
    
    Args:
        file_stream: File stream object
        max_size: Ukuran maksimum dalam bytes
        
    Returns:
        Tuple (is_valid, error_message)
    """
    try:
        # Cek ukuran file
        file_stream.seek(0, os.SEEK_END)
        file_size = file_stream.tell()
        file_stream.seek(0)  # Reset ke awal
        
        if file_size > max_size:
            max_mb = max_size / 1024 / 1024
            return False, f"File terlalu besar. Maksimum {max_mb:.1f}MB"
        
        if file_size == 0:
            return False, "File kosong"
        
        return True, None
        
    except Exception as e:
        logger.error(f"Error validating file size: {e}")
        return False, "Gagal memvalidasi ukuran file"


def calculate_file_hash(filepath: str) -> Optional[str]:
    """
    Hitung SHA256 hash dari file untuk tracking.
    
    Args:
        filepath: Path ke file
        
    Returns:
        SHA256 hash string atau None jika error
    """
    try:
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            # Baca file dalam chunks untuk efisiensi memory
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception as e:
        logger.error(f"Error calculating file hash: {e}")
        return None


def sanitize_query_string(query: str, max_length: int = 200) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Sanitasi dan validasi query string untuk pencarian.
    
    Args:
        query: Query string dari user
        max_length: Panjang maksimum query
        
    Returns:
        Tuple (is_valid, sanitized_query, error_message)
    """
    if not query:
        return False, None, "Query tidak boleh kosong"
    
    # Strip whitespace
    query = query.strip()
    
    # Validasi panjang minimum
    if len(query) < 2:
        return False, None, "Query terlalu pendek (minimum 2 karakter)"
    
    # Validasi panjang maksimum
    if len(query) > max_length:
        return False, None, f"Query terlalu panjang (maksimum {max_length} karakter)"
    
    # Hapus karakter berbahaya (basic sanitization)
    # Izinkan huruf, angka, spasi, dan beberapa karakter umum
    import re
    sanitized = re.sub(r'[^\w\s\-.,!?]', '', query, flags=re.UNICODE)
    
    if not sanitized:
        return False, None, "Query tidak valid setelah sanitasi"
    
    return True, sanitized, None


def validate_limit_parameter(limit: any, min_val: int = 1, max_val: int = 1000, default: int = 100) -> Tuple[int, Optional[str]]:
    """
    Validasi parameter limit untuk pagination/batching.
    
    Args:
        limit: Nilai limit dari user input
        min_val: Nilai minimum yang diizinkan
        max_val: Nilai maksimum yang diizinkan
        default: Nilai default jika invalid
        
    Returns:
        Tuple (validated_limit, error_message)
    """
    try:
        limit_int = int(limit)
        
        if limit_int < min_val or limit_int > max_val:
            return default, f"Limit harus antara {min_val}-{max_val}. Menggunakan default: {default}"
        
        return limit_int, None
        
    except (ValueError, TypeError):
        return default, f"Limit harus berupa angka. Menggunakan default: {default}"
