from dataclasses import dataclass

@dataclass
class SourceDetails:
    company_name: str
    document_type: str
    source_file: str
    start_page: int
    end_page: int

    @staticmethod
    def combineSources(src1: "SourceDetails", src2: "SourceDetails") -> "SourceDetails":
        if src1.source_file != src2.source_file:
            raise ValueError("Cannot combine sources from different files")

        # Ensure src1 starts first
        if src1.start_page > src2.start_page:
            src1, src2 = src2, src1

        # No overlap/touching
        if src1.end_page < src2.start_page:
            raise ValueError("Sources are not continuous")

        return SourceDetails(
            company_name=src1.company_name,
            document_type=src1.document_type,
            source_file=src1.source_file,
            start_page=src1.start_page,
            end_page=max(src1.end_page, src2.end_page),
        )

@dataclass
class RAGResponse:
    answer: str
    sources: list[SourceDetails]