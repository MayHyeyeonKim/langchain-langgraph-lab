from rest_framework import serializers


class ReviewRequestSerializer(serializers.Serializer):
    file = serializers.FileField(required=False, allow_empty_file=False)
    text = serializers.CharField(required=False, allow_blank=False)
    max_retries = serializers.IntegerField(required=False, min_value=1, max_value=5)

    def validate(self, attrs):
        if bool(attrs.get("file")) == bool(attrs.get("text")):
            raise serializers.ValidationError("Provide exactly one of 'file' or 'text'.")
        uploaded_file = attrs.get("file")
        if uploaded_file and not uploaded_file.name.lower().endswith(".pdf"):
            raise serializers.ValidationError({"file": "Only text-based PDF files are supported."})
        return attrs
