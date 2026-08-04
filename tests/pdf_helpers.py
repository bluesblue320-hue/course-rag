"""Helpers that build small valid PDFs for offline loader tests."""

from io import BytesIO

from pypdf import PdfWriter


def _object_bytes(object_number: int, body: bytes) -> bytes:
    return f"{object_number} 0 obj\n".encode() + body + b"\nendobj\n"


def make_text_pdf(pages: list[str]) -> bytes:
    """Build a minimal valid text PDF with one content stream per page."""
    page_count = len(pages)
    font_object = 3 + 2 * page_count
    kids = " ".join(f"{3 + index} 0 R" for index in range(page_count))

    contents: list[bytes] = [
        f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode() for text in pages
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}

    def write_object(object_number: int, body: bytes) -> None:
        offsets[object_number] = len(out)
        out.extend(_object_bytes(object_number, body))

    write_object(1, b"<< /Type /Catalog /Pages 2 0 R >>")
    write_object(
        2,
        f"<< /Type /Pages /Kids [{kids}] /Count {page_count} >>".encode(),
    )
    for index in range(page_count):
        write_object(
            3 + index,
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Contents {3 + page_count + index} 0 R "
                f"/Resources << /Font << /F1 {font_object} 0 R >> >> >>"
            ).encode(),
        )
    for index, content in enumerate(contents):
        write_object(
            3 + page_count + index,
            f"<< /Length {len(content)} >>\nstream\n".encode()
            + content
            + b"\nendstream",
        )
    write_object(
        font_object,
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    )

    xref_offset = len(out)
    total_objects = font_object + 1
    out.extend(f"xref\n0 {total_objects}\n".encode())
    out.extend(b"0000000000 65535 f \n")
    for object_number in range(1, total_objects):
        out.extend(f"{offsets[object_number]:010d} 00000 n \n".encode())
    out.extend(
        f"trailer\n<< /Size {total_objects} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF".encode()
    )
    return bytes(out)


def make_encrypted_pdf() -> bytes:
    """Build a password-encrypted PDF that cannot be read without a key."""
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.encrypt(user_password="secret-password", algorithm="RC4-128")
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()
