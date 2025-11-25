import os
from PIL import Image
import warnings

# 데이터 경로
DATA_DIR = "/home/yeonji/projects/opensource/분노"

print(f"'{DATA_DIR}' 검사 시작...")
bad_files = []

for root, dirs, files in os.walk(DATA_DIR):
    for filename in files:
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            file_path = os.path.join(root, filename)
            
            try:
                # 이미지를 열어서 확인 
                with Image.open(file_path) as img:
                    img.verify() 
                
                # 다시 열어서 실제 데이터 로드 시도 
                with Image.open(file_path) as img:
                    img.load()
                    
            except (IOError, SyntaxError, OSError) as e:
                print(f"망가진 파일 : {file_path}")
                bad_files.append(file_path)

print("-" * 30)
if bad_files:
    print(f" {len(bad_files)}개의 쓰레기 파일 발견")
    
    for bad_file in bad_files:
        try:
            os.remove(bad_file)
            print(f"  -> 완료: {bad_file}")
        except Exception as e:
            print(f"  -> 실패: {bad_file} ({e})")
else:
    print("모든 이미지가 정상")