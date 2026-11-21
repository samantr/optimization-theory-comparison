#!/usr/bin/env python3
"""Reproducible DOCX/PDF typesetting of the frozen academic report.

This script reads the exact existing Markdown manuscript. It does NOT run
calibration/optimizers or modify any experimental evidence.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DOCS=ROOT/"docs"
OUT=ROOT/"deliverables"
OUT.mkdir(parents=True,exist_ok=True)
SOURCE=DOCS/"ACADEMIC_REPORT_DRAFT.md"
BODY=OUT/"_report_for_typesetting.md"
WORD=OUT/"Optimization_Theory_GD_vs_Newton_Report.docx"
PDF=OUT/"Optimization_Theory_GD_vs_Newton_Report.pdf"


def main():
    source=SOURCE.read_text(encoding="utf-8")
    assert all(f"## {n}. " in source for n in range(1,10))
    assert "## Abstract" in source and "## References" in source
    source=source[source.index("## Abstract"):]
    if "### Editorial verification" in source:
        raise RuntimeError("Editorial notes leaked into manuscript.")
    # Pandoc silently drops \tag{n} in OOXML equations. Preserve equation
    # labels inside the display mathematics, where Word/LibreOffice render them.
    source=re.sub(r"\\tag\{(\d+)\}",lambda m:r"\qquad\mathrm{("+m.group(1)+")}",source)
    assert len(re.findall(r"\\mathrm\{\(\d+\)\}",source))==16
    # Convert the frozen evidence-derived SVGs for consistent Word/PDF support.
    import cairosvg
    for stem in ("primary_curvature_outcomes","dataset_paired_timing"):
        svg=DOCS/"figures"/f"{stem}.svg"
        png=DOCS/"figures"/f"{stem}.png"
        cairosvg.svg2png(url=str(svg),write_to=str(png),output_width=1800)
    source=source.replace("figures/primary_curvature_outcomes.svg","figures/primary_curvature_outcomes.png")
    source=source.replace("figures/dataset_paired_timing.svg","figures/dataset_paired_timing.png")
    BODY.write_text(source,encoding="utf-8")
    cmd=["pandoc",str(BODY),"-f","markdown+tex_math_dollars+raw_tex",
         "-t","docx","--resource-path",str(DOCS),
         "-o",str(WORD)]
    subprocess.run(cmd,check=True)
    from docx import Document
    from docx.shared import Inches,Pt,RGBColor,Mm
    from docx.enum.text import WD_ALIGN_PARAGRAPH,WD_BREAK
    from docx.enum.table import WD_TABLE_ALIGNMENT,WD_CELL_VERTICAL_ALIGNMENT
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    doc=Document(WORD)
    sec=doc.sections[0]
    # A4 is the university-standard paper size in Türkiye.
    sec.page_width=Mm(210)
    sec.page_height=Mm(297)
    sec.top_margin=Mm(22)
    sec.bottom_margin=Mm(20)
    sec.left_margin=Mm(22)
    sec.right_margin=Mm(22)
    sec.different_first_page_header_footer=True
    sec.header_distance=Inches(.35)
    sec.footer_distance=Inches(.35)
    styles=doc.styles
    normal=styles["Normal"]
    normal.font.name="Cambria"
    normal.font.size=Pt(10.5)
    normal.font.color.rgb=RGBColor(35,47,61)
    normal.paragraph_format.space_after=Pt(6)
    normal.paragraph_format.line_spacing=1.14
    for name,sz,space in [("Heading 1",14,12),("Heading 2",11.5,9)]:
        st=styles[name]
        st.font.name="Cambria"
        st.font.size=Pt(sz)
        st.font.bold=True
        st.font.color.rgb=RGBColor(19,54,92)
        st.paragraph_format.keep_with_next=True
        st.paragraph_format.space_before=Pt(space)
        st.paragraph_format.space_after=Pt(5)
    for p in doc.paragraphs:
        text=p.text.strip()
        if p.style.name=="Normal":
            p.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY
        if re.match(r"^(?:Table|Figure) \d+\.",text):
            p.alignment=WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.keep_with_next=True
            p.paragraph_format.space_before=Pt(6)
            p.paragraph_format.space_after=Pt(4)
            for run in p.runs:
                run.italic=True
                run.font.size=Pt(9)
        if text.startswith("Keywords:"):
            p.alignment=WD_ALIGN_PARAGRAPH.LEFT
    # Pandoc creates short automatic captions from image alt text; the
    # manuscript already contains more descriptive full figure captions.
    for p in list(doc.paragraphs):
        if p.style.name == "Image Caption" and p.text.startswith("Figure "):
            p._element.getparent().remove(p._element)
        if p.text.strip() == "References":
            p.paragraph_format.page_break_before=True
    # Pandoc handles OMML math. Retain all 16 numbered equations.
    for ti,table in enumerate(doc.tables):
        # Pandoc often supplies only its own "Table" style, not "Table Grid".
        table.style = "Table Grid" if "Table Grid" in [st.name for st in styles] else "Table"
        table.alignment=WD_TABLE_ALIGNMENT.CENTER
        table.autofit=True
        for ri,row in enumerate(table.rows):
            for cell in row.cells:
                cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
                for p in cell.paragraphs:
                    p.alignment=WD_ALIGN_PARAGRAPH.LEFT
                    p.paragraph_format.space_after=Pt(1)
                    for run in p.runs:
                        run.font.size=Pt(8.1 if len(table.columns)>3 else 8.5)
                        if ri==0:run.bold=True
            trPr=row._tr.get_or_add_trPr()
            cant=OxmlElement("w:cantSplit")
            trPr.append(cant)
            if ri==0:
                hdr=OxmlElement("w:tblHeader")
                hdr.set(qn("w:val"),"true")
                trPr.append(hdr)
    for shape in doc.inline_shapes:
        if shape.width>Inches(6.65):shape.width=Inches(6.65)
    # Add a dedicated front cover without replacing the exact report text.
    if not doc.paragraphs:raise RuntimeError("Conversion unexpectedly empty")
    first=doc.paragraphs[0]
    entries=[
        ("KOCAELI UNIVERSITY","uni"), 
        ("Graduate School of Natural and Applied Sciences","school"),
        ("Computer Engineering — Optimization Theory (BLM 599)","course"),
        ("","blank"),
        ("Comparative Analysis of Gradient Descent and Newton Optimization Methods in Nonlinear Regression Problems","title"),
        ("","blank"),
        ("Academic Course Project Report","subtitle"),
        ("","blank"),
        ("Prepared by","small"),
        ("Saman Tarighpeima","author"),
        ("","blank"),
        ("Instructor","small"),
        ("Prof. Dr. Yaşar Becerikli","author"),
        ("","blank"),
        ("November 2026","date"),
    ]
    for value,kind in entries:
        p=first.insert_paragraph_before(value)
        p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        if kind=="blank":p.paragraph_format.space_after=Pt(14)
        else:
            run=p.runs[0] if p.runs else p.add_run(value)
            run.font.name="Cambria"
            run.font.bold=kind in ("uni","title","subtitle","author")
            run.font.size=Pt(19 if kind=="title" else 15 if kind in ("uni","subtitle") else 12 if kind=="author" else 10)
            run.font.color.rgb=RGBColor(19,54,92) if kind in ("uni","title") else RGBColor(50,57,67)
            p.paragraph_format.space_after=Pt(13 if kind in ("title","uni") else 6)
    pb=first.insert_paragraph_before()
    pb.add_run().add_break(WD_BREAK.PAGE)
    # Running header and page number field; Word can refresh it.
    header=sec.header.paragraphs[0]
    header.text="KOCAELI UNIVERSITY  |  OPTIMIZATION THEORY"
    header.alignment=WD_ALIGN_PARAGRAPH.RIGHT
    for run in header.runs:
        run.font.name="Cambria";run.font.size=Pt(8);run.font.color.rgb=RGBColor(104,112,125)
    footer=sec.footer.paragraphs[0]
    footer.alignment=WD_ALIGN_PARAGRAPH.CENTER
    footer.add_run("Page ")
    run=footer.add_run()
    fld=OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"),"PAGE")
    run._r.addnext(fld)
    doc.core_properties.title="Comparative Analysis of Gradient Descent and Newton Optimization Methods in Nonlinear Regression Problems"
    doc.core_properties.author="Saman Tarighpeima"
    doc.core_properties.subject="Optimization Theory (BLM 599)"
    doc.save(WORD)
    result=subprocess.run(["libreoffice","-env:UserInstallation=file:///tmp/libreoffice-gd-newton-report",
       "--headless","--convert-to","pdf","--outdir",str(OUT),str(WORD)],
       check=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    if not PDF.exists():raise RuntimeError("LibreOffice failed to produce PDF: "+result.stdout)
    meta=subprocess.run(["pdfinfo",str(PDF)],capture_output=True,text=True,check=True).stdout
    pages=int(re.search(r"^Pages:\s*(\d+)",meta,re.M).group(1))
    if pages<10:raise RuntimeError(f"Only {pages} pages; 10 minimum required")
    pdftext=subprocess.run(["pdftotext","-layout",str(PDF),"-"],capture_output=True,text=True,check=True).stdout
    for key in ("Abstract","1. Introduction","9. Conclusion","References","Saman Tarighpeima"):
        if key not in pdftext:raise RuntimeError("Missing exported text: "+key)
    report=(OUT/"QA_SUMMARY.txt")
    report.write_text(
      f"Export: {WORD.name} and {PDF.name}\n"
      f"PDF pages: {pages}\n"
      f"Equation labels preserved as mathematical text: 1-16\n"
      f"Academic references: 21 (source manuscript audit)\n"
      f"Original optimization experiment: unchanged\n"
      f"Bibliography source-level QA and human visual review: STILL REQUIRED\n"
      f"Generated from: docs/ACADEMIC_REPORT_DRAFT.md\n",encoding="utf-8")
    print(report.read_text())
if __name__=="__main__":
    main()
