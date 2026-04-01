"""
Shared document formatting requirements for stage report prompts.
"""

DOCUMENT_FORMATTING_REQUIREMENTS = """DOCUMENT FORMATTING REQUIREMENTS (applies to all .tex outputs)
- Write in formal academic English suitable for sharing with research collaborators.
- Use complete paragraphs with clear topic sentences and logical flow.
- Use LaTeX sectioning (section, subsection, paragraph) for clear hierarchy.
- Use mathematical notation where appropriate (inline $...$ and display \\[...\\]).
- Include proper citations using \\cite{} with keys from references.bib.
- Use tables (tabular) for structured comparisons and itemize/enumerate for lists.
- The document must compile cleanly with tectonic.
- After writing the .tex file, compile it to PDF using tectonic.
"""

REPORT_FORMATTING_REQUIREMENTS = """REPORT FORMATTING REQUIREMENTS (applies to internal pipeline reports)
- Write in formal academic English suitable for sharing with research collaborators.
- Use complete paragraphs with clear topic sentences and logical flow.
- Use LaTeX sectioning (section, subsection, paragraph) for clear hierarchy.
- Use tables (tabular) for structured comparisons and itemize/enumerate for lists.
- The document must compile cleanly with tectonic.
- After writing the .tex file, compile it to PDF using tectonic.
"""
