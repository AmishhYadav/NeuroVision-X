// Shared helpers for the three model submission documents.
// Formatting: Times New Roman, black text only, A4, 1-inch margins, 12 pt body, 1.5 line spacing.
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell,
  WidthType, BorderStyle, ImageRun, PageBreak, TableOfContents, Footer, PageNumber, LevelFormat,
  SequentialIdentifier, TabStopType, VerticalAlign,
} = require("docx");
const { estimatePageNumbersWith } = require("docx/layout");

const FONT = "Times New Roman";
const ROOT = "/Users/amish/NeuroVision-X";
const FIG = path.join(__dirname, "figs");
const SEM = path.join(ROOT, "docs/research_docs/semester_report/figures");
const CONTENT_W = 9026; // A4 width 11906 - 2 x 1440 margins

// ---------------------------------------------------------------- helpers
function runs(text, opts = {}) {
  // Inline **bold** and *italic* markup.
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*)/g;
  let last = 0;
  let m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...opts }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, ...opts }));
    else out.push(new TextRun({ text: t.slice(1, -1), italics: true, ...opts }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...opts }));
  return out;
}
const P = (text, o = {}) => new Paragraph({ alignment: AlignmentType.JUSTIFIED, children: runs(text), ...o });
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: true, children: [new TextRun(t)] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const H3 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(t)] });
const B = (t, level = 0) => new Paragraph({ numbering: { reference: "bullets", level }, alignment: AlignmentType.JUSTIFIED, children: runs(t) });
let numListId = 0;
function NL(items) {
  const ref = `num${numListId++}`;
  numberingConfigs.push({
    reference: ref,
    levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 720, hanging: 360 } } } }],
  });
  return items.map((t) => new Paragraph({ numbering: { reference: ref, level: 0 }, alignment: AlignmentType.JUSTIFIED, children: runs(t) }));
}
const numberingConfigs = [];

function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}
function figure(file, caption, widthIn = 6.2) {
  const { w, h } = pngSize(file);
  const wpx = Math.round(widthIn * 96);
  const hpx = Math.round((wpx * h) / w);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 60 },
      children: [new ImageRun({ type: "png", data: fs.readFileSync(file), transformation: { width: wpx, height: hpx },
        altText: { title: caption, description: caption, name: path.basename(file) } })] }),
    new Paragraph({ style: "Caption", alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: "Figure ", bold: true }), new SequentialIdentifier("Figure"),
        new TextRun({ text: ". ", bold: true }), ...runs(caption)] }),
  ];
}

const border = { style: BorderStyle.SINGLE, size: 4, color: "000000" };
const borders = { top: border, bottom: border, left: border, right: border };
function table(caption, header, rows, widths) {
  const capPara = caption === null ? [] : null;
  const total = widths.reduce((a, b) => a + b, 0);
  const scale = CONTENT_W / total;
  const w = widths.map((x) => Math.floor(x * scale));
  w[w.length - 1] += CONTENT_W - w.reduce((a, b) => a + b, 0);
  const cell = (t, i, bold) => new TableCell({
    borders, width: { size: w[i], type: WidthType.DXA }, verticalAlign: VerticalAlign.CENTER,
    margins: { top: 50, bottom: 50, left: 90, right: 90 },
    children: [new Paragraph({ spacing: { line: 240, before: 0, after: 0 }, alignment: (i > 0 && /^[\d.,~+\u2212\-\u2013\s%\u00d7()\/*M]*$|^\u2014$|^default$/.test(t)) ? AlignmentType.CENTER : AlignmentType.LEFT,
      children: runs(t, { size: 20, bold: bold || undefined }) })],
  });
  return [
    ...(capPara || [new Paragraph({ style: "Caption", alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 200, after: 80 },
      children: [new TextRun({ text: "Table ", bold: true }), new SequentialIdentifier("Table"),
        new TextRun({ text: ". ", bold: true }), ...runs(caption)] })]),
    new Table({
      width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: w,
      rows: [new TableRow({ tableHeader: true, children: header.map((t, i) => cell(t, i, true)) }),
        ...rows.map((r) => new TableRow({ cantSplit: true, children: r.map((t, i) => cell(t, i, false)) }))],
    }),
    new Paragraph({ spacing: { before: 0, after: 120 }, children: [] }),
  ];
}
const blank = (n = 1) => Array.from({ length: n }, () => new Paragraph({ children: [] }));

const footer = new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
  children: [new TextRun({ children: [PageNumber.CURRENT], size: 20 })] })] });

function makeDoc(m) {
const doc = new Document({
  title: m.title,
  creator: "Amish Yadav",
  description: m.title,
  pageNumbers: estimatePageNumbersWith({ guess: true }),
  styles: {
    default: { document: { run: { font: FONT, size: 24, color: "000000" },
      paragraph: { spacing: { line: 360, after: 120 } } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 32, bold: true, color: "000000" },
        paragraph: { spacing: { before: 0, after: 240, line: 276 }, outlineLevel: 0, keepNext: true } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 28, bold: true, color: "000000" },
        paragraph: { spacing: { before: 240, after: 120, line: 276 }, outlineLevel: 1, keepNext: true } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 24, bold: true, italics: true, color: "000000" },
        paragraph: { spacing: { before: 200, after: 100, line: 276 }, outlineLevel: 2, keepNext: true } },
      { id: "Caption", name: "Caption", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 20, color: "000000" },
        paragraph: { spacing: { before: 60, after: 200, line: 260 } } },
      { id: "FrontTitle", name: "Front Title", basedOn: "Normal", next: "Normal",
        run: { font: FONT, size: 32, bold: true, color: "000000" },
        paragraph: { spacing: { before: 0, after: 240 } } },
      { id: "TOC1", name: "toc 1", basedOn: "Normal", run: { font: FONT, size: 24, bold: true }, paragraph: { spacing: { before: 120, after: 40, line: 276 } } },
      { id: "TOC2", name: "toc 2", basedOn: "Normal", run: { font: FONT, size: 24 }, paragraph: { indent: { left: 440 }, spacing: { before: 0, after: 40, line: 276 } } },
      { id: "TableofFigures", name: "table of figures", basedOn: "Normal", run: { font: FONT, size: 24 }, paragraph: { spacing: { before: 0, after: 60, line: 276 } } },
    ],
  },
  numbering: { config: [
    { reference: "bullets", levels: [
      { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
      { level: 1, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1440, hanging: 360 } } } },
    ] },
    ...numberingConfigs,
  ] },
  sections: [
    { properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
      children: m.titlePage },
    { properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
        pageNumbers: { start: 1, formatType: "lowerRoman" } } },
      footers: { default: footer },
      children: m.front },
    { properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
        pageNumbers: { start: 1, formatType: "decimal" } } },
      footers: { default: footer },
      children: m.body },
  ],
});
return doc;
}

function titlePage(m) {
  return [
    ...blank(2),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 160 }, children: [new TextRun({ text: m.kicker, bold: true, size: 28 })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 360 }, children: [new TextRun({ text: m.period, italics: true, size: 24 })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: [new TextRun({ text: "NeuroVision-X", bold: true, size: 44 })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120, line: 360 }, children: [new TextRun({ text: m.subtitle, size: 30 })] }),
    ...blank(3),
    new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Submitted by", size: 24, italics: true })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 }, children: [new TextRun({ text: "Amish Yadav", bold: true, size: 28 })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Course Project, Semester 2026", size: 24 })] }),
    ...blank(4),
    new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: m.date, size: 24 })] }),
    ...blank(2),
    new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Research software. Not a medical device. Not for clinical use.", italics: true, size: 22 })] }),
  ];
}
function frontMatter(abstractParas, abbrevRows) {
  return [
    new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: true, children: [new TextRun("Abstract")] }),
    ...abstractParas,
    new Paragraph({ style: "FrontTitle", pageBreakBefore: true, children: [new TextRun("Table of Contents")] }),
    new TableOfContents("Table of Contents", { hyperlink: true, headingStyleRange: "1-2" }),
    new Paragraph({ style: "FrontTitle", pageBreakBefore: true, children: [new TextRun("List of Figures")] }),
    new TableOfContents("List of Figures", { hyperlink: true, captionLabelIncludingNumbers: "Figure" }),
    new Paragraph({ style: "FrontTitle", pageBreakBefore: true, children: [new TextRun("List of Tables")] }),
    new TableOfContents("List of Tables", { hyperlink: true, captionLabelIncludingNumbers: "Table" }),
    new Paragraph({ style: "FrontTitle", pageBreakBefore: true, children: [new TextRun("List of Abbreviations")] }),
    ...table(null, ["Abbreviation", "Meaning"], abbrevRows, [2.2, 6.8]),
  ];
}
const REF = (t) => new Paragraph({ alignment: AlignmentType.LEFT, indent: { left: 567, hanging: 567 }, spacing: { line: 276, after: 100 }, children: runs(t) });
const QUOTE = (t) => new Paragraph({ alignment: AlignmentType.JUSTIFIED, indent: { left: 720, right: 720 }, spacing: { before: 120, after: 160 }, children: runs(t) });
const fig = (name) => path.join(FIG, name + ".png");
const sem = (name) => path.join(SEM, name + ".png");
function write(doc, out) {
  return Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(out, buf); console.log("wrote", out); });
}
module.exports = { P, H1, H2, H3, B, NL, figure, table, blank, runs, REF, QUOTE, fig, sem, ROOT, titlePage, frontMatter, makeDoc, write, Paragraph, TextRun, AlignmentType, path };
