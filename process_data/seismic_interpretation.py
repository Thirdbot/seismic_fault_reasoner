import argparse
import json
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import PictureItem, TableItem
from tqdm import tqdm


class SeismicInterpretationDataBuilder:
    def __init__(self):
        self.root = Path(__file__).resolve().parent.parent
        self.output_root = Path(__file__).resolve().parent / "interpretation"
        self.reports_dir = self.root / "data" / "download" / "reports" / "Reports" / "data"
        self.pdf_paths = [
            self.reports_dir / "Statoil internal report on Smeaheia Subsurface 2016 - selected extracts.pdf",
            self.reports_dir / "Troll_kystnaer_subsurface_status_report final_Gassnova.pdf",
        ]
        self.extracted_output_path = self.output_root / "extracted_data"
        self.extracted_output_path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _converter() -> DocumentConverter:
        options = PdfPipelineOptions()
        options.images_scale = 2.0
        options.generate_picture_images = True
        options.generate_table_images = True
        return DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
        )

    @staticmethod
    def _page_and_bbox(element):
        if not getattr(element, "prov", None):
            return None, None

        prov = element.prov[0]
        bbox = getattr(prov, "bbox", None)
        if bbox is not None:
            try:
                bbox = bbox.model_dump()
            except Exception:
                bbox = str(bbox)
        return getattr(prov, "page_no", None), bbox

    @staticmethod
    def _text(element) -> str:
        for attr in ("text", "orig", "content"):
            value = getattr(element, attr, None)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    @staticmethod
    def _caption(element, document) -> str:
        try:
            return (element.caption_text(document) or "").strip()
        except Exception:
            return ""

    @staticmethod
    def _is_heading(element, text: str) -> bool:
        name = type(element).__name__.lower()
        return "section" in name or "heading" in name or "title" in name or (text[:2].isdigit() and len(text) < 120)

    def _document_items(self, document):
        items = []
        chunks = []
        section_path = []

        for index, (element, level) in enumerate(document.iterate_items()):
            page, bbox = self._page_and_bbox(element)
            text = self._text(element)

            if text and self._is_heading(element, text):
                level_int = int(level) if isinstance(level, int) else len(section_path)
                section_path = [text] if level_int <= 0 else [*section_path[:level_int], text]

            record = {
                "index": index,
                "element": element,
                "page": page,
                "bbox": bbox,
                "text": text,
                "section_path": list(section_path),
                "self_ref": getattr(element, "self_ref", None),
            }
            items.append(record)

            if text and not isinstance(element, (PictureItem, TableItem)):
                chunk_id = f"txt_{len(chunks) + 1:05d}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "type": "text",
                    "page": page,
                    "bbox": bbox,
                    "section_path": list(section_path),
                    "text": text,
                    "docling_self_ref": getattr(element, "self_ref", None),
                    "source_index": index,
                })
                record["chunk_id"] = chunk_id

        return items, chunks

    @staticmethod
    def _context(items, source_index, page, window=4):
        def useful(item):
            return item.get("text") and not isinstance(item["element"], (PictureItem, TableItem))

        before = [
            {"chunk_id": item.get("chunk_id"), "page": item.get("page"), "text": item["text"]}
            for item in reversed(items[:source_index])
            if useful(item)
        ][:window]
        before.reverse()

        after = [
            {"chunk_id": item.get("chunk_id"), "page": item.get("page"), "text": item["text"]}
            for item in items[source_index + 1:]
            if useful(item)
        ][:window]

        same_page_text = "\n".join(item["text"] for item in items if item.get("page") == page and useful(item))
        linked_chunk_ids = [item["chunk_id"] for item in before + after if item.get("chunk_id")]
        return {
            "nearby_text_before": before,
            "nearby_text_after": after,
            "same_page_text": same_page_text,
            "linked_chunk_ids": linked_chunk_ids,
        }

    def _write_jsonl(self, path: Path, records: list[dict]) -> None:
        with path.open("w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def extract_data(self, pdf_paths=None) -> None:
        converter = self._converter()
        for pdf in tqdm(pdf_paths or self.pdf_paths):
            if not pdf.exists():
                raise FileNotFoundError(f"PDF not found: {pdf}")

            doc_id = pdf.stem
            doc_dir = self.output_root / doc_id
            image_dir = doc_dir / "images"
            image_dir.mkdir(parents=True, exist_ok=True)

            document = converter.convert(pdf).document
            (self.extracted_output_path / f"{doc_id}.md").write_text(
                document.export_to_markdown(),
                encoding="utf-8",
            )

            items, chunks = self._document_items(document)
            for chunk in chunks:
                chunk["doc_id"] = doc_id

            media_records = []
            counts = {"figure": 0, "table": 0, "skipped": 0}
            for item in tqdm(items, desc=f"Media: {doc_id}"):
                element = item["element"]
                if not isinstance(element, (PictureItem, TableItem)):
                    continue

                caption = self._caption(element, document)
                if not caption:
                    counts["skipped"] += 1
                    continue

                image = element.get_image(document)
                if image is None:
                    counts["skipped"] += 1
                    continue

                media_type = "figure" if isinstance(element, PictureItem) else "table"
                if media_type == "figure" and (image.size[0] < 100 or image.size[1] < 80):
                    counts["skipped"] += 1
                    continue

                counts[media_type] += 1
                element_id = f"{'fig' if media_type == 'figure' else 'table'}_{counts[media_type]:04d}"
                image_path = image_dir / f"{element_id}.png"
                image.save(image_path)

                media_records.append({
                    "doc_id": doc_id,
                    "element_id": element_id,
                    "docling_self_ref": item["self_ref"],
                    "type": media_type,
                    "page": item["page"],
                    "bbox": item["bbox"],
                    "section_path": item["section_path"],
                    "caption": caption,
                    "image_path": image_path.as_posix(),
                    **self._context(items, item["index"], item["page"]),
                })

            self._write_jsonl(doc_dir / "chunks.jsonl", chunks)
            self._write_jsonl(doc_dir / "manifest.jsonl", media_records)

            print(f"\nPDF: {pdf.name}")
            print(f"Saved figures: {counts['figure']}")
            print(f"Saved tables: {counts['table']}")
            print(f"Skipped media: {counts['skipped']}")
            print(f"Chunks: {doc_dir / 'chunks.jsonl'}")
            print(f"Manifest: {doc_dir / 'manifest.jsonl'}")


seismic_interpretation_DataBuider = SeismicInterpretationDataBuilder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract report text, figures, and tables for interpretation.")
    parser.add_argument("--pdf", type=Path, action="append", help="PDF path to extract. Defaults to bundled reports.")
    args = parser.parse_args()

    builder = SeismicInterpretationDataBuilder()
    builder.extract_data(args.pdf)
