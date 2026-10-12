from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

pdf_path = "rag.pdf"

pipeline_options = PdfPipelineOptions()

pipeline_options.do_ocr = True
pipeline_options.do_chart_extraction = False

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options
        )
    }
)

result = converter.convert(pdf_path)

markdown = result.document.export_to_markdown()

with open("parsed_rag.md", "w", encoding="utf-8") as f:
    f.write(markdown)

print("Parsing completed!")
print("Saved to: parsed_rag.md")