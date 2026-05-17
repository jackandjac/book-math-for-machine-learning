#!/usr/bin/env node
// Protect math blocks from marked's italic/em parsing.
// Strategy: two-pass approach using placeholders to prevent double-matching.
// Usage: node preprocess-math.js <input.md> <output.md>

const fs = require('fs');

const [,, inputFile, outputFile] = process.argv;
if (!inputFile || !outputFile) {
  console.error('Usage: node preprocess-math.js <input.md> <output.md>');
  process.exit(1);
}

let content = fs.readFileSync(inputFile, 'utf8');

// Pass 1: Extract ALL display math $$ ... $$ into a store, replace with placeholders
const displayStore = [];
content = content.replace(/\$\$([\s\S]+?)\$\$/g, (_match, latex) => {
  const id = displayStore.length;
  displayStore.push(latex);
  return `DISPLAYMATH_PLACEHOLDER_${id}_END`;
});

// Pass 2: Extract ALL inline math $ ... $ into a store, replace with placeholders
const inlineStore = [];
content = content.replace(/\$([^$\n]+?)\$/g, (_match, latex) => {
  const id = inlineStore.length;
  inlineStore.push(latex);
  return `INLINEMATH_PLACEHOLDER_${id}_END`;
});

// Pass 3: Restore display math as raw HTML blocks (marked passes raw HTML verbatim)
displayStore.forEach((latex, id) => {
  content = content.replace(
    `DISPLAYMATH_PLACEHOLDER_${id}_END`,
    `\n<div class="math-block">$$${latex}$$</div>\n`
  );
});

// Pass 4: Restore inline math as raw HTML spans
inlineStore.forEach((latex, id) => {
  content = content.replace(
    `INLINEMATH_PLACEHOLDER_${id}_END`,
    `<span class="math-inline">$${latex}$</span>`
  );
});

fs.writeFileSync(outputFile, content, 'utf8');
console.error(`Math protected: wrote ${outputFile} (${displayStore.length} display, ${inlineStore.length} inline blocks)`);
