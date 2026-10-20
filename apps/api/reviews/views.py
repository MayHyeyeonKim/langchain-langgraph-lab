from tempfile import NamedTemporaryFile, TemporaryDirectory

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from document_reviewer.documents import DocumentExtractionError, extract_pdf_text
from document_reviewer.graph import ReviewWorkflow
from document_reviewer.rag import (
    LocalRagStore,
    RagUnavailableError,
    ReferenceDocument,
    build_context,
    source_references,
)

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
            references = self._read_references(serializer.validated_data.get("reference_files", []))
            context = ""
            sources = []
            if references:
                with TemporaryDirectory() as rag_directory:
                    store = LocalRagStore(persist_directory=rag_directory)
                    store.index(references)
                    chunks = store.retrieve(document)
                    context = build_context(chunks)
                    sources = source_references(chunks)
            result = ReviewWorkflow(max_retries=serializer.validated_data.get("max_retries")).run(
                document, reference_context=context, sources=sources
            )
        except DocumentExtractionError as error:
            return Response({"detail": str(error)}, status=status.HTTP_422_UNPROCESSABLE_ENTITY)
        except RagUnavailableError as error:
            return Response({"detail": str(error)}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
        except Exception as error:  # noqa: BLE001 - adapter boundary translates local LLM failures to HTTP.
            return Response({"detail": f"Review workflow failed: {error}"}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
        return Response(result.model_dump(), status=status.HTTP_200_OK)

    @staticmethod
    def _read_references(files) -> list[ReferenceDocument]:
        references: list[ReferenceDocument] = []
        for uploaded_file in files:
            with NamedTemporaryFile(suffix=".pdf") as temp_file:
                for chunk in uploaded_file.chunks():
                    temp_file.write(chunk)
                temp_file.flush()
                references.append(
                    ReferenceDocument(text=extract_pdf_text(temp_file.name), source=uploaded_file.name)
                )
        return references
