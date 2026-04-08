import pandas as pd
import psycopg2
from sentence_transformers import SentenceTransformer
import numpy as np
from pgvector.psycopg2 import register_vector

DB_CONFIG = {
    "host": "localhost",
    "database": "Databaseexport", 
    "user": "postgres",
    "password": "102207",
    "port": "5432" 
}

print("--- Đang tải model AI (BGE-M3)... ---")
model = SentenceTransformer('BAAI/bge-m3')

def clean_val(val):
    """Biến NaN thành chuỗi rỗng"""
    if pd.isna(val):
        return ""
    return str(val).strip()

try:
    df = pd.read_excel('DatabaseExport.xlsx')
    conn = psycopg2.connect(**DB_CONFIG)
    register_vector(conn)  # Quan trọng: đăng ký vector type
    cur = conn.cursor()

    print(f"--- Bắt đầu xử lý {len(df)} dòng ---")

    for index, row in df.iterrows():
        ten_hang = clean_val(row['Tên hàng hóa'])
        stt = row['STT'] if pd.notna(row['STT']) else index + 1

        if not ten_hang:
            print(f"⚠️ Dòng {index + 1}: Bỏ qua vì Tên hàng trống.")
            continue

        # Encode tên hàng thành vector
        vector_ten_hang = model.encode(ten_hang).tolist()

        # Dùng stt làm khóa chính (an toàn hơn)
        sql = """
            UPDATE thong_tin_hang_hoa 
            SET vt_ten_hang = %s 
            WHERE stt = %s
        """
        
        cur.execute(sql, (vector_ten_hang, stt))

        if (index + 1) % 50 == 0:
            print(f"✅ Đã xong {index + 1} dòng...")
            conn.commit()

    conn.commit()
    print("--- 🎉 XONG! Vector cho tên hàng đã được tạo! ---")

except Exception as e:
    print(f"❌ Lỗi: {e}")
    conn.rollback()
finally:
    if 'cur' in locals(): cur.close()
    if 'conn' in locals(): conn.close()