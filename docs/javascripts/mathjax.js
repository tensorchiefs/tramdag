window.MathJax = {
  loader: { load: ["[tex]/boldsymbol"] },
  tex: {
    packages: { "[+]": ["boldsymbol"] },
    inlineMath: [["\\(", "\\)"], ["$", "$"]],
    displayMath: [["\\[", "\\]"], ["$$", "$$"]],
    processEscapes: true,
  },
  // No ignoreHtmlClass/processHtmlClass pair here, on purpose. The Material
  // idiom ("ignore everything, process .arithmatex") only reaches math that
  // sits DIRECTLY inside the marked element, because every descendant matches
  // the ignore pattern again. That is true of an arithmatex span, and false of
  // a notebook: mkdocs-jupyter renders the markdown cells itself, so their
  // math sits six divs below the .jupyter-wrapper and was never typeset.
  // MathJax's default skipHtmlTags already keeps it out of code, pre, script
  // and style, which is where notebook source and output live.
  options: {},
};
// the first document$ emission fires before the MathJax script (loaded after
// this file) exists — guard, and let the CDN's own startup typeset handle the
// initial render
document$.subscribe(() => {
  if (window.MathJax && MathJax.typesetPromise) {
    MathJax.typesetClear();
    MathJax.texReset();
    MathJax.typesetPromise();
  }
});
