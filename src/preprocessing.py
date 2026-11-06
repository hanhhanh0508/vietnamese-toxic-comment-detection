"""
Module Preprocessing cho Đồ án Phát hiện Bình luận Độc hại Tiếng Việt
Phụ trách: Sinh viên 1 (Pre-processing Lead)
Nhiệm vụ:
- Chuẩn hóa Unicode NFC
- Regex loại bỏ URL, Mention (@username), email
- Rút gọn ký tự lặp (Repeated characters)
- Ánh xạ từ điển Teencode (data/dictionaries/teencode.json)
- Hỗ trợ ánh xạ Emoji (data/dictionaries/emoji.json)
- Tách từ tiếng Việt (Word Segmentation qua pyvi hoặc underthesea)
"""

import os
import re
import json
import unicodedata
from typing import Dict, Optional

# Thử import công cụ tách từ tiếng Việt
HAS_UNDERTHESEA = False
try:
    from underthesea import word_tokenize
    HAS_UNDERTHESEA = True
except Exception:
    HAS_UNDERTHESEA = False

HAS_PYVI = False
try:
    from pyvi import ViTokenizer
    HAS_PYVI = True
except Exception:
    HAS_PYVI = False


def load_dictionary(file_path: str) -> Dict[str, str]:
    """Đọc từ điển ánh xạ từ file JSON"""
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# Khởi tạo mặc định nếu đường dẫn tồn tại
_DEFAULT_TEENCODE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "dictionaries", "teencode.json"
)
_DEFAULT_EMOJI_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "dictionaries", "emoji.json"
)

GLOBAL_TEENCODE_DICT: Dict[str, str] = load_dictionary(_DEFAULT_TEENCODE_PATH)
GLOBAL_EMOJI_DICT: Dict[str, str] = load_dictionary(_DEFAULT_EMOJI_PATH)


def normalize_unicode(text: str) -> str:
    """
    Đưa văn bản tiếng Việt về chuẩn Unicode NFC (dựng sẵn)
    Tránh lỗi mâu thuẫn giữa Unicode NFD (tổ hợp) và NFC
    """
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize("NFC", text)


def remove_urls_and_mentions(text: str) -> str:
    """
    Loại bỏ các đường dẫn web (URL), email và thẻ tag người dùng (@username)
    """
    if not isinstance(text, str):
        return ""
    # Xóa URL http / https / www
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text, flags=re.MULTILINE)
    # Xóa email
    text = re.sub(r'\S+@\S+\.\S+', ' ', text)
    # Xóa tag người dùng (@username)
    text = re.sub(r'@[\w\._-]+', ' ', text)
    return text


def clean_repeated_characters(text: str) -> str:
    """
    Rút gọn các ký tự kéo dài lặp liên tiếp >= 3 lần về 1 hoặc 2 ký tự
    Ví dụ: 'ngonnnnn' -> 'ngon', 'vclllll' -> 'vcl', 'quáaaaa' -> 'quá'
    """
    if not isinstance(text, str):
        return ""
    # Thu gọn ký tự lặp từ 3 lần trở lên về 1 ký tự
    return re.sub(r'(.)\1{2,}', r'\1', text)


def convert_emoji(text: str, emoji_dict: Optional[Dict[str, str]] = None) -> str:
    """
    Ánh xạ các biểu tượng cảm xúc (Emoji) thành chuỗi tiếng Việt mô tả
    Ví dụ: '😡' -> ' phẫn nộ '
    """
    if not isinstance(text, str):
        return ""
    mapping = emoji_dict if emoji_dict is not None else GLOBAL_EMOJI_DICT
    for emoji_char, replacement in mapping.items():
        if emoji_char in text:
            text = text.replace(emoji_char, f" {replacement.strip()} ")
    return text


def normalize_teencode(text: str, teencode_dict: Optional[Dict[str, str]] = None) -> str:
    """
    Chuẩn hóa các từ viết tắt, từ lóng, teencode dựa trên từ điển ánh xạ
    Ví dụ: 'k' -> 'không', 'vcl' -> 'vãi lờ', 'dc' -> 'được'
    """
    if not isinstance(text, str):
        return ""
    mapping = teencode_dict if teencode_dict is not None else GLOBAL_TEENCODE_DICT
    if not mapping:
        return text

    words = text.split()
    normalized_words = []
    for word in words:
        lower_word = word.lower()
        # Tách các dấu câu dính liền ở đầu/cuối nếu có
        match = re.match(r'^([^\w]*)(.*?)([^\w]*)$', lower_word)
        if match:
            prefix, core, suffix = match.groups()
            if core in mapping:
                rep = mapping[core]
                normalized_words.append(f"{prefix}{rep}{suffix}")
            else:
                normalized_words.append(word)
        else:
            normalized_words.append(mapping.get(lower_word, word))

    return " ".join(normalized_words)


def word_segment(text: str, engine: str = "underthesea") -> str:
    """
    Tách từ tiếng Việt, nối các từ ghép bằng dấu gạch dưới '_'
    Cần thiết cho quá trình Tokenize của PhoBERT
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    if engine == "underthesea" and HAS_UNDERTHESEA:
        return word_tokenize(text, format="text")
    elif engine == "pyvi" and HAS_PYVI:
        return ViTokenizer.tokenize(text)
    elif HAS_UNDERTHESEA:
        return word_tokenize(text, format="text")
    elif HAS_PYVI:
        return ViTokenizer.tokenize(text)
    else:
        # Fallback nếu chưa cài thư viện
        return text


def preprocess_text(
    text: str,
    teencode_dict: Optional[Dict[str, str]] = None,
    emoji_dict: Optional[Dict[str, str]] = None,
    segment_words: bool = True,
    word_engine: str = "underthesea"
) -> str:
    """
    Hàm tổng hợp toàn bộ Pipeline tiền xử lý văn bản tiếng Việt
    
    Quy trình:
    1. Đưa về Unicode chuẩn NFC
    2. Xóa URL, tag @username, email
    3. Ánh xạ biểu tượng Emoji thành từ mô tả tiếng Việt
    4. Rút gọn ký tự lặp kéo dài
    5. Ánh xạ teencode & từ lóng về từ gốc
    6. Dọn dẹp ký tự thừa và khoảng trắng
    7. Tách từ tiếng Việt (nối từ ghép bằng '_')
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Bước 1: Unicode NFC
    text = normalize_unicode(text)

    # Bước 2: Regex URL & Tag @mentions
    text = remove_urls_and_mentions(text)

    # Bước 3: Emoji
    text = convert_emoji(text, emoji_dict)

    # Bước 4: Ký tự lặp
    text = clean_repeated_characters(text)

    # Bước 5: Teencode
    text = normalize_teencode(text, teencode_dict)

    # Bước 6: Dọn khoảng trắng thừa và ký tự đặc biệt vô nghĩa
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # Bước 7: Tách từ tiếng Việt
    if segment_words:
        text = word_segment(text, engine=word_engine)

    return text


# --- Kiểm thử nhanh module khi chạy trực tiếp ---
if __name__ == "__main__":
    test_cases = [
        "Sp nàylk ngonnn vcl 😡😡, mn ko nên mua nha!! https://shop-lua-dao.vn @admin",
        "Đm tụi mài cút về nước đi, đồ súc vật 🤬🤬",
        "Hàng dùng bthg, giao nhanh, tks shop nhìu nhé ❤️❤️",
        "hôm nay trời đẹpppp wáaaaa, đi chơiii k mn?"
    ]
    
    print("=== KIỂM THỬ MODULE PREPROCESSING ===")
    print(f"- Số từ trong từ điển Teencode: {len(GLOBAL_TEENCODE_DICT)}")
    print(f"- Số emoji trong từ điển Emoji: {len(GLOBAL_EMOJI_DICT)}")
    print(f"- Thư viện tách từ: pyvi ({HAS_PYVI}), underthesea ({HAS_UNDERTHESEA})\n")
    
    for i, raw in enumerate(test_cases, 1):
        processed = preprocess_text(raw, segment_words=True)
        print(f"[{i}] GỐC : {raw}")
        print(f"    SẠCH: {processed}\n")
