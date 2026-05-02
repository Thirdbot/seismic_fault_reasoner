import json

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
import fitz
from pathlib import Path

from docling_core.types.doc import PictureItem, TableItem
from tqdm import tqdm


class seismic_interpretation_DataBuider:
    def __init__(self):
        self.root = Path(__file__).parent.parent.absolute()
        self.parent = Path(__file__).parent.absolute()
        self.data_path = self.root / 'data' / 'download'
        self.report_path = self.data_path / 'reports' / 'Reports' / 'data'
        self.pdf1 = self.report_path / 'Statoil internal report on Smeaheia Subsurface 2016 - selected extracts.pdf'
        self.pdf2 = self.report_path / 'Troll_kystnaer_subsurface_status_report final_Gassnova.pdf'
        self.converter = DocumentConverter()
        self.image_path = self.parent / 'interpretation'
        self.image_path.mkdir(parents=True, exist_ok=True)
        self.extracted_output_path = self.parent / 'interpretation' / "extracted_data"
        self.extracted_output_path.mkdir(parents=True, exist_ok=True)

    def safe_get_page_and_bbox(self,element):
        if not getattr(element, "prov", None):
            return None, None

        prov = element.prov[0]
        page_no = getattr(prov, "page_no", None)

        bbox = None
        if getattr(prov, "bbox", None) is not None:
            try:
                bbox = prov.bbox.model_dump()
            except Exception:
                bbox = str(prov.bbox)

        return page_no, bbox

    def safe_get_text(self,element):
        for attr in ["text", "orig", "content"]:
            value = getattr(element, attr, None)
            if isinstance(value, str) and value.strip():
                return value.strip()

        return ""

    def safe_get_caption(self,element, document):
        try:
            caption = element.caption_text(document)
            if caption:
                return caption.strip()
        except Exception:
            pass

        return ""

    def looks_like_heading(self,element, text):
        name = type(element).__name__.lower()

        if "section" in name or "heading" in name or "title" in name:
            return True

        # Backup heuristic for numbered report sections
        if text[:2].isdigit() and len(text) < 120:
            return True

        return False

    def extract_data(self):
        pipeline_options = PdfPipelineOptions()
        pipeline_options.images_scale = 2.0
        pipeline_options.generate_picture_images = True
        pipeline_options.generate_table_images = True  # if your version supports it

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        for pdf in tqdm([self.pdf1, self.pdf2]):
            doc = self.converter.convert(source=pdf)
            markdown = doc.document.export_to_markdown()
            (self.extracted_output_path / f"{pdf.name.replace('.pdf', '.md')}").write_text(markdown,encoding="utf-8")

            sub_image_path = (self.image_path / f"{pdf.name.removesuffix('.pdf')}") / 'images'
            manifest_path =  (self.image_path / f"{pdf.name.removesuffix('.pdf')}") / 'manifest.jsonl'
            chunks_path = (self.image_path / f"{pdf.name.removesuffix('.pdf')}") / "chunks.jsonl"
            sub_image_path.mkdir(parents=True, exist_ok=True)
            doc_id = pdf.stem

            result = converter.convert(pdf)
            document = result.document
            markdown = document.export_to_markdown()

            self.extracted_output_path.mkdir(parents=True, exist_ok=True)
            (self.extracted_output_path / f"{doc_id}.md").write_text(
                markdown,
                encoding="utf-8"
            )

            # Flatten document items once
            linear_items = []
            text_chunks = []
            section_path = []
            text_count = 0

            for idx, (element, level) in enumerate(document.iterate_items()):
                page_no, bbox = self.safe_get_page_and_bbox(element)
                text = self.safe_get_text(element)
                item_type = type(element).__name__

                if text and self.looks_like_heading(element, text):
                    # Simple section tracking
                    level_int = int(level) if isinstance(level, int) else len(section_path)

                    if level_int <= 0:
                        section_path = [text]
                    else:
                        section_path = section_path[:level_int]
                        section_path.append(text)

                item_record = {
                    "index": idx,
                    "element": element,
                    "type": item_type,
                    "page": page_no,
                    "bbox": bbox,
                    "text": text,
                    "section_path": list(section_path),
                    "self_ref": getattr(element, "self_ref", None),
                }

                linear_items.append(item_record)

                # Save normal text chunks separately
                if text and not isinstance(element, (PictureItem, TableItem)):
                    text_count += 1
                    chunk_id = f"txt_{text_count:05d}"

                    chunk_record = {
                        "doc_id": doc_id,
                        "chunk_id": chunk_id,
                        "type": "text",
                        "page": page_no,
                        "bbox": bbox,
                        "section_path": list(section_path),
                        "text": text,
                        "docling_self_ref": getattr(element, "self_ref", None),
                        "source_index": idx,
                    }

                    text_chunks.append(chunk_record)
                    item_record["chunk_id"] = chunk_id

            def get_nearby_context(source_index, page_no, window=4):
                before = []
                after = []

                # previous useful text chunks
                for item in reversed(linear_items[:source_index]):
                    if item.get("text") and not isinstance(item["element"], (PictureItem, TableItem)):
                        before.append({
                            "chunk_id": item.get("chunk_id"),
                            "page": item.get("page"),
                            "text": item["text"],
                        })
                    if len(before) >= window:
                        break

                before = list(reversed(before))

                # next useful text chunks
                for item in linear_items[source_index + 1:]:
                    if item.get("text") and not isinstance(item["element"], (PictureItem, TableItem)):
                        after.append({
                            "chunk_id": item.get("chunk_id"),
                            "page": item.get("page"),
                            "text": item["text"],
                        })
                    if len(after) >= window:
                        break

                # same page text
                same_page = []
                for item in linear_items:
                    if item.get("page") == page_no and item.get("text"):
                        if not isinstance(item["element"], (PictureItem, TableItem)):
                            same_page.append(item["text"])

                same_page_text = "".join(same_page)

                linked_chunk_ids = []
                for ctx in before + after:
                    if ctx.get("chunk_id"):
                        linked_chunk_ids.append(ctx["chunk_id"])

                return {
                    "nearby_text_before": before,
                    "nearby_text_after": after,
                    "same_page_text": same_page_text,
                    "linked_chunk_ids": linked_chunk_ids,
                }

            media_records = []
            figure_count = 0
            table_count = 0
            skipped_no_caption = 0

            for item in tqdm(linear_items, desc=f"Media: {doc_id}"):
                element = item["element"]

                if isinstance(element, PictureItem):
                    caption = self.safe_get_caption(element, document)

                    # Main logo filter:
                    # if there is no caption, do not save it.
                    if not caption:
                        skipped_no_caption += 1
                        continue

                    image = element.get_image(document)
                    if image is None:
                        continue

                    # Optional small-logo/icon filter
                    w, h = image.size
                    if w < 100 or h < 80:
                        skipped_no_caption += 1
                        continue

                    figure_count += 1
                    element_id = f"fig_{figure_count:04d}"
                    image_path = sub_image_path / f"{element_id}.png"
                    image.save(image_path)

                    context = get_nearby_context(
                        source_index=item["index"],
                        page_no=item["page"],
                        window=4
                    )

                    media_records.append({
                        "doc_id": doc_id,
                        "element_id": element_id,
                        "docling_self_ref": item["self_ref"],
                        "type": "figure",
                        "page": item["page"],
                        "bbox": item["bbox"],
                        "section_path": item["section_path"],
                        "caption": caption,
                        "image_path": str(image_path),
                        **context,
                    })

                elif isinstance(element, TableItem):
                    caption = self.safe_get_caption(element, document)

                    # You can choose whether to keep captionless tables.
                    # For now, same rule: skip if no caption.
                    if not caption:
                        skipped_no_caption += 1
                        continue

                    image = element.get_image(document)
                    if image is None:
                        continue

                    table_count += 1
                    element_id = f"table_{table_count:04d}"
                    image_path = sub_image_path / f"{element_id}.png"
                    image.save(image_path)

                    context = get_nearby_context(
                        source_index=item["index"],
                        page_no=item["page"],
                        window=4
                    )

                    media_records.append({
                        "doc_id": doc_id,
                        "element_id": element_id,
                        "docling_self_ref": item["self_ref"],
                        "type": "table",
                        "page": item["page"],
                        "bbox": item["bbox"],
                        "section_path": item["section_path"],
                        "caption": caption,
                        "image_path": str(image_path),
                        **context,
                    })

            # Save text chunks
            with chunks_path.open("w", encoding="utf-8") as f:
                for record in text_chunks:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

            # Save figure/table manifest
            with manifest_path.open("w", encoding="utf-8") as f:
                for record in media_records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

            print(f"\nPDF: {pdf.name}")
            print(f"Saved figures: {figure_count}")
            print(f"Saved tables: {table_count}")
            print(f"Skipped no-caption/small images: {skipped_no_caption}")
            print(f"Chunks: {chunks_path}")
            print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    seismic_interpretation = seismic_interpretation_DataBuider()
    seismic_interpretation.extract_data()