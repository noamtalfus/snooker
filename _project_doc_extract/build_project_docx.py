import html
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
TEXT_PATH = ROOT / "_project_doc_extract" / "docx_text.txt"
OUT_PATH = ROOT / "project_document_resend.docx"


TITLE_LINES = {
    "פרויקט בהנדסת תוכנה",
    "בינה מלאכותית",
    "סוכן AI למשחק Reversi",
}

HEADING_LINES = {
    "תוכן העניינים",
    "מבוא",
    "תיאור המשחק",
    "כללי",
    "כללי המשחק",
    "מטרת המשחק – כיצד מנצחים ?",
    "נתונים נוספים",
    "מדריך למשתמש",
    "מודל סביבה – סוכן",
    "תיאור המודל באופן כללי",
    "מימוש המודל בפרויקט",
    "המחלקה State",
    "המחלקה Graphics",
    "המחלקה Reversi – Environment",
    "המחלקה Human_Agent",
    "המחלקה Random_Agent",
    "המחלקה Game",
    "סוכן Reinforcement - DQN",
    "למידת חיזוק",
    "מודל MDP – כללי",
    "האלגוריתם Q-Learning",
    "למידת חיזוק עמוקה DQN",
    "האלגוריתם DQN",
    "Replay buffer",
    "מימוש DQN",
    "תוצאות ומסקנות המחקר",
    "רפליקציה",
    "תודות",
    "נספחים",
}


def paragraph(text, style=None, align="right", bold=False):
    escaped = html.escape(text)
    style_xml = f'<w:pStyle w:val="{style}"/>' if style else ""
    bold_xml = "<w:b/>" if bold else ""
    size = "44" if style == "Title" else "32" if style == "Heading1" else "28" if style == "Heading2" else "24"
    return f"""
    <w:p>
      <w:pPr>
        {style_xml}
        <w:bidi/>
        <w:jc w:val="{align}"/>
        <w:spacing w:after="160"/>
      </w:pPr>
      <w:r>
        <w:rPr>
          <w:rFonts w:ascii="Arial" w:hAnsi="Arial" w:cs="Arial"/>
          <w:rtl/>
          {bold_xml}
          <w:sz w:val="{size}"/>
          <w:szCs w:val="{size}"/>
        </w:rPr>
        <w:t>{escaped}</w:t>
      </w:r>
    </w:p>
    """


def build_document_xml(lines):
    body_parts = []
    for i, line in enumerate(lines):
        text = line.strip()
        if not text:
            body_parts.append("<w:p/>")
            continue
        if text in TITLE_LINES:
            body_parts.append(paragraph(text, "Title", "center", True))
        elif text in HEADING_LINES:
            style = "Heading1" if text in {"תוכן העניינים", "מבוא", "תיאור המשחק", "מדריך למשתמש", "מודל סביבה – סוכן", "סוכן Reinforcement - DQN", "תוצאות ומסקנות המחקר", "רפליקציה", "תודות", "נספחים"} else "Heading2"
            body_parts.append(paragraph(text, style, "right", True))
        elif i < 8 and ":" in text:
            body_parts.append(paragraph(text, None, "center", False))
        else:
            body_parts.append(paragraph(text))

    section = """
    <w:sectPr>
      <w:pgSz w:w="12240" w:h="15840"/>
      <w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="720" w:footer="720" w:gutter="0"/>
      <w:bidi/>
    </w:sectPr>
    """
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:wpc="http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas"
  xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006"
  xmlns:o="urn:schemas-microsoft-com:office:office"
  xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
  xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"
  xmlns:v="urn:schemas-microsoft-com:vml"
  xmlns:wp14="http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing"
  xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
  xmlns:w10="urn:schemas-microsoft-com:office:word"
  xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
  xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml"
  xmlns:wpg="http://schemas.microsoft.com/office/word/2010/wordprocessingGroup"
  xmlns:wpi="http://schemas.microsoft.com/office/word/2010/wordprocessingInk"
  xmlns:wne="http://schemas.microsoft.com/office/word/2006/wordml"
  xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape"
  mc:Ignorable="w14 wp14">
  <w:body>
    {''.join(body_parts)}
    {section}
  </w:body>
</w:document>
"""


def styles_xml():
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:pPr><w:bidi/><w:jc w:val="right"/><w:spacing w:after="160"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Arial" w:hAnsi="Arial" w:cs="Arial"/><w:sz w:val="24"/><w:szCs w:val="24"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Title">
    <w:name w:val="Title"/>
    <w:basedOn w:val="Normal"/>
    <w:pPr><w:bidi/><w:jc w:val="center"/><w:spacing w:after="220"/></w:pPr>
    <w:rPr><w:b/><w:rFonts w:ascii="Arial" w:hAnsi="Arial" w:cs="Arial"/><w:sz w:val="44"/><w:szCs w:val="44"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="heading 1"/>
    <w:basedOn w:val="Normal"/>
    <w:pPr><w:bidi/><w:jc w:val="right"/><w:spacing w:before="240" w:after="120"/></w:pPr>
    <w:rPr><w:b/><w:rFonts w:ascii="Arial" w:hAnsi="Arial" w:cs="Arial"/><w:sz w:val="32"/><w:szCs w:val="32"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading2">
    <w:name w:val="heading 2"/>
    <w:basedOn w:val="Normal"/>
    <w:pPr><w:bidi/><w:jc w:val="right"/><w:spacing w:before="180" w:after="100"/></w:pPr>
    <w:rPr><w:b/><w:rFonts w:ascii="Arial" w:hAnsi="Arial" w:cs="Arial"/><w:sz w:val="28"/><w:szCs w:val="28"/></w:rPr>
  </w:style>
</w:styles>
"""


def main():
    lines = TEXT_PATH.read_text(encoding="utf-8").splitlines()
    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>
"""
    rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>
"""
    doc_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>
"""
    app = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
  xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Codex</Application>
</Properties>
"""
    core = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
  xmlns:dc="http://purl.org/dc/elements/1.1/"
  xmlns:dcterms="http://purl.org/dc/terms/"
  xmlns:dcmitype="http://purl.org/dc/dcmitype/"
  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>פרויקט בהנדסת תוכנה</dc:title>
  <dc:creator>Codex</dc:creator>
</cp:coreProperties>
"""

    with zipfile.ZipFile(OUT_PATH, "w", zipfile.ZIP_DEFLATED) as docx:
        docx.writestr("[Content_Types].xml", content_types)
        docx.writestr("_rels/.rels", rels)
        docx.writestr("word/_rels/document.xml.rels", doc_rels)
        docx.writestr("word/document.xml", build_document_xml(lines))
        docx.writestr("word/styles.xml", styles_xml())
        docx.writestr("docProps/app.xml", app)
        docx.writestr("docProps/core.xml", core)

    print(OUT_PATH)


if __name__ == "__main__":
    main()
