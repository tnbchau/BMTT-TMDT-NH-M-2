import os
import pandas as pd
import numpy as np

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "emails_merged.csv") # Đổi tên cho đúng ý nghĩa merge
EXCLUDE_FILES = ["phishing_email.csv"]

def load_and_standardize(file_path):
    # 1. ĐỌC DỮ LIỆU
    print(f"--> Đang đọc: {os.path.basename(file_path)}")
    df = pd.read_csv(file_path, encoding='utf-8')

    # 2. CHUẨN HÓA CỘT
    # Mapping các tên cột khác nhau về chuẩn chung
    df.columns = [c.lower() for c in df.columns]   
    # Schema chuẩn
    schema = ["sender", "date", "subject", "body", "urls", "label"]    
    # Đảm bảo đủ cột, thiếu thì điền NaN
    for col in schema:
        if col not in df.columns:
            df[col] = np.nan
    df = df[schema].copy()

    # 3. CHUẨN HÓA NHÃN (LABEL)
    def clean_label(val):
        str_val = str(val).lower().strip()
        if str_val in ["1", "1.0", "phishing", "spam"]:
            return 1
        return 0
    
    df["label"] = df["label"].apply(clean_label)

    # 4. TẠO TEXT TỔNG HỢP 
    # Xử lý điền khuyết bằng chuỗi rỗng trước khi cộng chuỗi
    df["combined_text"] = (
        df["subject"].fillna("") + " " + df["body"].fillna("")
    ).str.strip()

    # Lọc bỏ dòng rác (không có nội dung gì)
    initial_len = len(df)
    df = df[df["combined_text"].str.len() > 1].reset_index(drop=True)
    if len(df) < initial_len:
        print(f"    🧹 Đã lọc bỏ {initial_len - len(df)} dòng rỗng.")

    return df

def main():
    if not os.path.exists(RAW_DIR):
        print(f"❌ Lỗi: Không tìm thấy thư mục {RAW_DIR}")
        return

    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)

    all_dfs = []
    
    # Duyệt file
    for file_name in os.listdir(RAW_DIR):
        if file_name in EXCLUDE_FILES:
            print(f"⏩ Bỏ qua (theo yêu cầu): {file_name}")
            continue
        if file_name.endswith(".csv"):
            file_path = os.path.join(RAW_DIR, file_name)
            try:
                df = load_and_standardize(file_path)
                all_dfs.append(df)
            except Exception as e:
                print(f"❌ Lỗi nghiêm trọng khi xử lý {file_name}: {e}")

    if all_dfs:
        merged_df = pd.concat(all_dfs, ignore_index=True)
        
        # Lưu file
        merged_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig') # utf-8-sig để Excel mở không lỗi font
        
        print("\n" + "="*30)
        print("KẾT QUẢ GỘP DỮ LIỆU:")
        print(f"✅ Đã lưu tại: {OUTPUT_FILE}")
        print(f"📊 Tổng số mẫu: {len(merged_df)}")
        print(f"⚠️ Tỷ lệ Phishing: {merged_df['label'].mean():.2%}")
        print("="*30)
    else:
        print("⚠️ Không có dữ liệu nào được gộp.")

if __name__ == "__main__":
    main()