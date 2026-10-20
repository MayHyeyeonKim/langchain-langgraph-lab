from rest_framework import serializers # DRF(Django REST Framework)에서 데이터를 변환하고 검증하는 역할 가져옴


class ReviewRequestSerializer(serializers.Serializer): # Serializer를 상속받아 리뷰 요청을 검증하는 클래스
    file = serializers.FileField(required=False, allow_empty_file=False) # 파일을 받고
    text = serializers.CharField(required=False, allow_blank=False) # 텍스트를 받음
    max_retries = serializers.IntegerField(required=False, min_value=1, max_value=5) # 최대 재시도 횟수
    reference_files = serializers.ListField(child=serializers.FileField(), required=False) # 참고 파일들    

    def validate(self, attrs): # 현재 객체(self)에서 입력 데이터(attrs) 전체를 검증하는 메서드를 정의
        if bool(attrs.get("file")) == bool(attrs.get("text")):
            raise serializers.ValidationError("Provide exactly one of 'file' or 'text'.")
        uploaded_file = attrs.get("file")
        if uploaded_file and not uploaded_file.name.lower().endswith(".pdf"):
            raise serializers.ValidationError({"file": "Only text-based PDF files are supported."})
        for reference_file in attrs.get("reference_files", []):
            if not reference_file.name.lower().endswith(".pdf"): # 업로드된 파일이 있는데, 파일명이 .pdf로 끝나지 않으면,
                raise serializers.ValidationError( # 검증 오류를 발생 시켜라
                    {"reference_files": "Reference files must be text-based PDFs."}
                )
        return attrs

# raise = Python의 throw, except = JavaScript의 catch