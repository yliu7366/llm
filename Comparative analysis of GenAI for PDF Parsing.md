### **A Comparative Analysis of Local and Cloud-Based Generative AI Models for Complex PDF Document Parsing**

---

#### **Executive Summary**

This paper details a systematic investigation into the efficacy of various Generative AI models for the complex task of parsing unstructured PDF documents into semantically coherent, structured text. The study evaluates a spectrum of open-source models executed on local hardware against a state-of-the-art commercial API, exploring both vision-language and language-only tasks. Initial methodologies involving local models, even those with up to 70 billion parameters, consistently failed to handle the combined complexity of non-standard layouts and hierarchical instruction sets, exhibiting failure modes such as prompt overload, procedural hallucination, and resource exhaustion. In contrast, a two-stage pipeline utilizing a frontier commercial model, Gemini 2.5 Pro, proved to be a robust and effective solution. The first stage employed the model's vision-language capabilities for high-fidelity text and layout extraction. The second stage utilized its language-only reasoning to perform a sophisticated semantic refactoring of the extracted text, successfully relocating embedded figures and tables to create a clean, readable narrative. The findings conclude that for tasks requiring a high degree of layout analysis, instruction following, and procedural reasoning, a significant capability gap exists between current local open-source models and state-of-the-art commercial APIs.

---

#### **1. Introduction**

The conversion of PDF documents, particularly complex, multi-column academic manuscripts, into structured, machine-readable text is a significant challenge in data processing. Traditional Optical Character Recognition (OCR) methods often fail to interpret logical layouts, resulting in jumbled text that loses its original reading order and semantic structure. The advent of large-scale, multimodal Generative AI (GenAI) models presents a new paradigm for solving this problem.

This investigation was undertaken to systematically evaluate the performance of various GenAI models on a complex PDF parsing task. The primary objective was to determine if currently available open-source models, run on local commodity hardware, possess the necessary capabilities to match the performance of a state-of-the-art commercial API. The evaluation encompasses two primary phases: an initial vision-language parsing task to extract content from the PDF, and a subsequent language-only refactoring task to semantically restructure the extracted text.

#### **2. Methodology**

The study was conducted on a hardware testbed equipped with dual 24GB VRAM GPUs and 256GB of system memory, utilizing the Ollama framework for local model execution. A single, representative academic paper featuring a non-standard layout (an abstract block spanning the full width of the left column and extending partially into the right column, followed by a two-column body) was used as the benchmark document.

**2.1. Baseline Establishment: OCR**
An initial baseline was established using the Tesseract OCR engine. A high-resolution (300 DPI) image of the document page was rendered and processed by Tesseract to produce a raw text output. This served as a benchmark for basic character recognition and layout handling without semantic understanding.

**2.2. Phase 1: Vision-Language Parsing**
This phase focused on the end-to-end task of converting the PDF page image into structured markdown text.

*   **Local Model Testing:** A wide array of open-source vision-language models were systematically tested, including `llava:34b`, `llama3.2-vision:11b`, `Llama 4 Scout`, and `Llama 4 Maverick`. The testing process involved iterative prompt engineering and parameter tuning (`temperature`, `top_p`, `num_ctx`) to elicit the best possible performance from each model.

*   **Commercial API Testing:** The same task was performed using the Gemini 2.5 Pro model via its official API. A sophisticated, hierarchical prompt was developed to instruct the model to perform high-fidelity transcription, handle the complex layout, and filter out metadata such as page headers, footers, and footnotes.

**2.3. Phase 2: Semantic Refactoring**
This phase addressed a limitation observed in the visually accurate output from Phase 1, where figure captions and tables were embedded mid-paragraph, disrupting the narrative flow. The task was to use an LLM to refactor this "dirty" markdown into a clean, readable document by relocating these elements to an appendix.

*   **Local Model Testing:** The `llama3:8b` and `llama3:70b` language models were tested. They were provided with a detailed, multi-step procedural prompt instructing them to identify, tag, move, and re-assemble the content.

*   **Commercial API Testing:** The Gemini 2.5 Pro model was given the same procedural prompt and the "dirty" markdown text to perform the cleanup task.

#### **3. Results and Analysis**

**3.1. OCR Baseline Performance**
The Tesseract OCR pipeline successfully extracted the majority of characters but failed fundamentally on layout. It mixed text from adjacent columns, destroying the logical reading order. Furthermore, it exhibited poor accuracy on off-baseline characters like superscripts used for citations.

**3.2. Local Vision-Language Model Performance**
All tested local models failed to complete the vision-parsing task successfully. The failures manifested in several distinct modes:
*   **Layout Navigation Failure:** Smaller models like `llama3.2-vision:11b` could transcribe initial sections but consistently failed at the column break, entering a state of repetitive looping or hallucination.
*   **Prompt Overload:** Larger models like `llava:34b` exhibited complete model collapse when given a complex, hierarchical prompt. They appeared to ignore the visual input entirely and generated generic, templated text.
*   **Task Misinterpretation:** `Llama 4 Scout`, despite being a newer model, refused the transcription task, instead generating creative but factually incorrect text based on its general understanding of the topic.
*   **Hardware Limitation:** `Llama 4 Maverick`, a large Mixture-of-Experts (MoE) model, exceeded the 24GB VRAM capacity of the testbed hardware, resulting in a `cudaMalloc: out of memory` error. This proved the task was not runnable on the available hardware.
*   **Parameter Tuning Ineffectiveness:** Adjusting parameters such as the context window (`num_ctx`) did not solve the core issues and, in some cases, led to earlier and more severe failures by overwhelming the model's attention mechanism.

**3.3. Commercial API Vision-Language Model Performance**
The Gemini 2.5 Pro model, using the `PUBLICATION_PROMPT`, successfully executed all instructions. It flawlessly transcribed the text, correctly navigated the complex full-width-to-two-column layout, and accurately filtered out all specified metadata (headers, footers, footnotes). The resulting output was a visually accurate markdown representation of the document's core content.

**3.4. Semantic Refactoring Performance**
This phase revealed a critical gap in procedural reasoning.
*   **Local Model Failure:** Both the `llama3:8b` and `llama3:70b` models failed to execute the multi-step refactoring procedure. Instead of following the instructions, they adopted the *persona* of an "expert technical editor" and defaulted to the simpler, more common task of writing a review of the provided text. The failure was identical across both model sizes, indicating a limitation in agency, not just scale.
*   **Commercial API Success:** Gemini 2.5 Pro correctly interpreted and executed the complex, multi-step `RESTRUCTURING_PROMPT`. It successfully identified, tagged, and relocated all figure captions and tables to an appendix, producing a clean, semantically coherent final document.

#### **4. Discussion**

The results indicate a significant capability gap between the tested open-source models and the state-of-the-art commercial API for this class of task. The failure of local models can be attributed to several factors:

*   **Limited Procedural Reasoning:** The primary failure mode, especially in the refactoring task, was an inability to follow a complex, stateful algorithm described in natural language. The models defaulted to a simpler, persona-driven behavior, indicating a weakness in agentic reasoning.
*   **Architectural Brittleness:** The difficulty local models had with complex layouts and hierarchical prompts suggests their architecture, while powerful for text generation, is less robust for tasks requiring simultaneous high-level vision analysis and deep instruction following.
*   **Scale and Training:** The success of Gemini 2.5 Pro is likely attributable to its massive scale (trillion-plus parameters via MoE) and its training on a vast corpus of data that includes complex reasoning and procedural tasks. This allows it to handle a level of prompt complexity and nuance that overwhelms smaller, monolithic models.

#### **5. Conclusion**

For the task of parsing complex academic documents into a clean, structured, and semantically coherent format, a multi-stage pipeline utilizing a frontier commercial model like Gemini 2.5 Pro is the only viable and robust solution identified in this investigation. The initial vision-language pass provides high-fidelity, layout-aware transcription, while a subsequent language-only pass performs the sophisticated semantic refactoring that local models are currently unable to execute reliably.

While local open-source models continue to advance rapidly, this study demonstrates that a significant gap remains in their ability to perform complex, multi-step, agentic tasks that require deep procedural reasoning. For mission-critical document processing workflows demanding high accuracy and reliability, leveraging state-of-the-art commercial APIs remains the necessary approach.

---

***Disclaimer:*** *Generative AI was utilized to assist in the summarization and structuring of the experimental findings documented in this paper.*

---

### **Supplement: Prompts Used**

**STRICT_PROMPT**

Your task is to act as a document parsing engine.
Analyze the provided image of a document page and convert its content into a single, clean markdown file.

Follow these rules strictly:
1.  Your response must contain ONLY the markdown content of the page.
2.  Do NOT include any introductory sentences, preamble, or explanations like "Certainly, here is the markdown..." or "Here is the content extracted...".
3.  Your output must begin directly with the first element of the document page (e.g., a heading like '# Title', a list item, or plain text).
4.  Preserve the original structure, including headings, lists, tables, and paragraphs.

**MASTER_PROMPT**

Your task is to act as a precision document parsing engine.
Analyze the provided image of a document page and convert its content into a single, clean markdown file.

Follow these rules with absolute precision:
1.  **Transcribe Verbatim:** Your primary goal is to transcribe the text exactly as it appears. Do not add, omit, or summarize any information.
2.  **Handle Multi-Column Layouts:** The document may have multiple columns. After you finish transcribing the first column, you must continue transcribing from the top of the second column, and so on for any subsequent columns.
3.  **No External Knowledge:** Do NOT add any sections, phrases, or sentences that are not explicitly present in the image. Do not generate 'Acknowledgements' or any other common sections unless they are visually present.
4.  **Strictly Markdown:** Your output must contain ONLY the markdown content of the page. Do NOT include any preamble, explanations, or closing remarks. Your response must begin directly with the first text element on the page.

**NAVIGATOR_PROMPT**

Transcribe all text from the image into markdown.
For documents with two columns, you must finish the entire left column before you start the right column.

**BEST_EFFORT_PROMPT**

Your task is to transcribe the text from the image into markdown. Follow these rules exactly.

**Rules:**
1.  **Transcribe the main text.**
2.  **Layout:** For multi-column text, finish the entire left column before starting the right column.
3.  **DO NOT INCLUDE:**
    - The single line of text at the very top of the page (header).
    - The page number at the very bottom of the page (footer).
    - The footnote section at the bottom of the first column, which is often separated by a line.

Your output must be only the markdown text. Do not write any explanations or apologies.

**PUBLICATION_PROMPT**

Your task is to act as an expert academic manuscript parser. Your goal is to extract only the main narrative body of the paper.

Analyze the provided image of a document page and convert its core content into a single, clean markdown file, following these hierarchical rules:

**Primary Goal: Extract the Core Manuscript**
- Focus exclusively on the main text, including the abstract, introduction, methods, results, discussion, and conclusion.
- The final output should represent a clean, readable version of the paper's narrative content.

**Exclusion Rules: What to Ignore**
1.  **Ignore Page Headers and Footers:** Explicitly exclude repeating elements at the very top (running titles, author names) and bottom (page numbers, journal names, DOIs) of the page.
2.  **Exclude First-Page Footnotes:** On the first page, identify and completely ignore the footnote section typically found at the bottom of the left column. This section contains author affiliations, correspondence details, keywords, and disclosure statements and is NOT part of the main manuscript body. It is often separated by a horizontal line.

**Layout and Formatting Rules**
- After processing any full-width elements (like the title and abstract), you MUST return to the top of the leftmost column and process it fully before moving to the next column.
- Transcribe the main text verbatim. Do not add, summarize, or omit any of the core manuscript content.
- Your response must contain ONLY the cleaned markdown content. Do not include any conversational preamble or notes.

**RESTRUCTURING_PROMPT**

Your task is to act as an expert technical editor. You will be given a markdown text that was accurately transcribed from a scientific paper but preserves the original, sometimes awkward, layout. Your goal is to refactor this text into a clean, semantically coherent document with an uninterrupted narrative flow.

Follow this three-step process precisely:

**Step 1: Identify and Tag Structural Elements**
- Read through the entire text and identify all figure captions (e.g., text starting with "Figure 1:", "Fig. 2.", etc.) and all markdown tables.
- For each element you find, "tag" it by replacing it with a unique, simple placeholder. For example:
    - Replace the first figure caption with `[FIGURE_1_HERE]`
    - Replace the second figure caption with `[FIGURE_2_HERE]`
    - Replace the first table with `[TABLE_1_HERE]`

**Step 2: Create a Clean Narrative Body**
- After tagging and removing all figures and tables, the remaining text should be the main narrative body of the paper.
- Ensure this narrative flows correctly without any semantic gaps. Paragraphs that were previously interrupted should now be seamlessly joined.

**Step 3: Relocate and Consolidate Extracted Elements**
- At the very end of the document, create a new section titled '## Appendix: Figures and Tables'.
- Under this heading, list all the figure captions and tables you extracted in Step 1, in the order they appeared in the original text. For example:
    - **Figure 1:** [Full text of the first figure caption]
    - **Figure 2:** [Full text of the second figure caption]
    - **Table 1:** [The full markdown table]

Your final output must be a single, clean markdown document. Do not include any explanations or commentary about your process.
