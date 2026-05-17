#!/usr/bin/env node
// Preprocess markdown for md-to-pdf:
//   1. Extract headings and build a clickable HTML Table of Contents
//   2. Add anchor IDs to every heading so TOC links work in PDF
//   3. Protect math blocks from marked's italic/em parsing
//
// Usage: node preprocess-math.js <input.md> <output.md>

const fs = require('fs');

const [,, inputFile, outputFile] = process.argv;
if (!inputFile || !outputFile) {
  console.error('Usage: node preprocess-math.js <input.md> <output.md>');
  process.exit(1);
}

let content = fs.readFileSync(inputFile, 'utf8');

// ── Step 1: Extract headings ──────────────────────────────────────────────────
// Match # / ## / ### headings (not inside code blocks)
const headings = [];
const lines = content.split('\n');
let inCodeBlock = false;

for (const line of lines) {
  if (line.startsWith('```')) { inCodeBlock = !inCodeBlock; continue; }
  if (inCodeBlock) continue;
  const m = line.match(/^(#{1,3})\s+(.+)$/);
  if (m) {
    const level = m[1].length;
    const text = m[2].trim();
    // Strip inline math placeholders / markdown bold/italic for display text
    const displayText = text.replace(/\$[^$]+\$/g, '').replace(/[*_`]/g, '').trim();
    // GitHub-style slug: lowercase, spaces→hyphens, strip non-alphanumeric except hyphens
    const slug = text.toLowerCase().replace(/[^\w\s-]/g, '').replace(/\s+/g, '-').replace(/-+/g, '-').replace(/^-|-$/g, '');
    headings.push({ level, text, displayText, slug });
  }
}

// ── Step 2: Build TOC HTML ────────────────────────────────────────────────────
function buildToc(headings) {
  let toc = `<div class="toc-wrapper">
<h1 class="toc-title" style="page-break-before:avoid;">Table of Contents</h1>
<div class="toc">
`;
  for (const h of headings) {
    const indent = h.level === 1 ? 0 : h.level === 2 ? 1 : 2;
    const cls = `toc-h${h.level}`;
    toc += `<div class="toc-entry ${cls}" style="padding-left:${indent * 1.4}em">` +
           `<a href="#${h.slug}">${h.displayText}</a></div>\n`;
  }
  toc += `</div>\n</div>\n\n<div style="page-break-after:always;"></div>\n\n`;
  return toc;
}

const tocHtml = buildToc(headings);

// ── Step 3: Add anchor IDs to headings in content ────────────────────────────
// Replace each heading with an HTML anchor + the original heading
// Track slug usage to handle duplicates
const slugCount = {};

inCodeBlock = false;
const outLines = [];
for (const line of lines) {
  if (line.startsWith('```')) { inCodeBlock = !inCodeBlock; outLines.push(line); continue; }
  if (inCodeBlock) { outLines.push(line); continue; }

  const m = line.match(/^(#{1,3})\s+(.+)$/);
  if (m) {
    const text = m[2].trim();
    let slug = text.toLowerCase().replace(/[^\w\s-]/g, '').replace(/\s+/g, '-').replace(/-+/g, '-').replace(/^-|-$/g, '');
    // Handle duplicate slugs
    if (slugCount[slug] !== undefined) {
      slugCount[slug]++;
      slug = `${slug}-${slugCount[slug]}`;
    } else {
      slugCount[slug] = 0;
    }
    // Inject anchor before heading (raw HTML, passed through by marked)
    outLines.push(`<a id="${slug}"></a>`);
    outLines.push(line);
  } else {
    outLines.push(line);
  }
}

content = outLines.join('\n');

// ── Step 4: Protect math from marked ─────────────────────────────────────────
// Pass 1: display math $$...$$ → placeholder
const displayStore = [];
content = content.replace(/\$\$([\s\S]+?)\$\$/g, (_match, latex) => {
  const id = displayStore.length;
  displayStore.push(latex);
  return `DISPLAYMATH_PLACEHOLDER_${id}_END`;
});

// Pass 2: inline math $...$ → placeholder
const inlineStore = [];
content = content.replace(/\$([^$\n]+?)\$/g, (_match, latex) => {
  const id = inlineStore.length;
  inlineStore.push(latex);
  return `INLINEMATH_PLACEHOLDER_${id}_END`;
});

// Pass 3: restore display math as raw HTML div
// Use a function replacer to prevent $ in latex from being interpreted as backreferences
displayStore.forEach((latex, id) => {
  content = content.replace(
    `DISPLAYMATH_PLACEHOLDER_${id}_END`,
    () => `\n<div class="math-block">$$${latex}$$</div>\n`
  );
});

// Pass 4: restore inline math as raw HTML span
// Use a function replacer to prevent $ in latex from being interpreted as backreferences
inlineStore.forEach((latex, id) => {
  content = content.replace(
    `INLINEMATH_PLACEHOLDER_${id}_END`,
    () => `<span class="math-inline">$${latex}$</span>`
  );
});

// ── Step 5: Prepend TOC ───────────────────────────────────────────────────────
content = tocHtml + content;

fs.writeFileSync(outputFile, content, 'utf8');
console.error(`Done: ${headings.length} headings, ${displayStore.length} display math, ${inlineStore.length} inline math`);
