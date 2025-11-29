import classifier 

# 테스트할 이미지 파일명
IMAGE_PATH = "/home/yeonji/projects/opensource/happy_test2.jpg" 

try:
    print(f"'{IMAGE_PATH}' 파일을 읽는 중...")
    
    with open(IMAGE_PATH, "rb") as f:
        image_bytes = f.read()
    
    print("예측 함수 실행...")
    
    result = classifier.predict_image(image_bytes)
    
    print("\n" + "="*30)
    print("결과 확인")
    print("="*30)
    print(f"감정: {result.get('emotion')}")
    print(f"확신도: {result.get('confidence')}")

except FileNotFoundError:
    print(f"오류: '{IMAGE_PATH}' 파일을 찾을 수 없습니다. 같은 폴더에 사진을 넣어주세요.")