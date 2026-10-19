from tempfile import NamedTemporaryFile

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from document_reviewer.documents import DocumentExtractionError, extract_pdf_text
from document_reviewer.graph import ReviewWorkflow

from .serializers import ReviewRequestSerializer


class ReviewDocumentView(APIView):
    """Synchronous local endpoint; use a job queue before production-scale documents."""

    def post(self, request):
        serializer = ReviewRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        try:
            if "text" in serializer.validated_data:
                document = serializer.validated_data["text"]
            else:
                upload = serializer.validated_data["file"]
                with NamedTemporaryFile(suffix=".pdf") as temp_file:
                    for chunk in upload.chunks():
                        temp_file.write(chunk)
                    temp_file.flush()
                    document = extract_pdf_text(temp_file.name)
            result = ReviewWorkflow(max_retries=serializer.validated_data.get("max_retries")).run(document)
        except DocumentExtractionError as error:
            return Response({"detail": str(error)}, status=status.HTTP_422_UNPROCESSABLE_ENTITY)
        except Exception as error:  # noqa: BLE001 - adapter boundary translates local LLM failures to HTTP.
            return Response({"detail": f"Review workflow failed: {error}"}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
        return Response(result.model_dump(), status=status.HTTP_200_OK)
