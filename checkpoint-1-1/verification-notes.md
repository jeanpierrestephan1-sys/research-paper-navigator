# Checkpoint 1.1 — Verification Notes


---

## Prompt 1 of 5 — Counterfeit Conundrum: main finding + quote

Prompt: In the paper 'The Counterfeit Conundrum: Can Code Language Models Grasp
the Nuances of Their Incorrect Generations?' (Gu et al., ACL 2024), what is the
main finding about how code models perceive their own incorrect generations?
Quote the relevant sentence.

Ground truth: "most models have a very shallow understanding of counterfeits...
models mistakenly classify them as correct." (DBLP_conf_acl_GuLJOLSS24.pdf)

Run 1
    Response: "can generally recognize their own incorrect generations"
              Quote: "Our results show that code LMs can largely perceive their
              own incorrect generations..."  — ABSENT from paper
    Observations: Disobeyed the system prompt, did not say it lacked access, answered confidently instead.
                  A hallucination. Confident response, even with a quote and
                  bridge to do more analysis and responses.
                  I did not find that text in the paper, it wasn't there. It
                  was fabricated.

Run 2
    Response: "generally do not recognize their own incorrect generations well"
              Quote: "We find that code LMs are often unable to distinguish
              their own correct generations from counterfeit ones." — ABSENT
    Observations: Disobeyed the system prompt, did not say it lacked access, answered confidently instead.
                  Inconsistent response by completely contradicting itself, now
                  aligned to the paper's finding.

Run 3
    Response: "do not reliably recognize their own incorrect generations"
              Quote: "We find that code models struggle to discern their own
              erroneous outputs..." — ABSENT
    Observations: Disobeyed the system prompt, did not say it lacked access, answered confidently instead.
                  New fabrication, inconsistent from the prior one, yet still aligned to the paper's finding but not cited.
                  Must assume response was a result of model training rather than the document. 

Run 4
    Response: "do not reliably recognize their own incorrect code generations"
              Quote: "We find that code language models have limited ability to
              perceive their own incorrect generations." — ABSENT
    Observations: Disobeyed the system prompt, did not say it lacked access, answered confidently instead.
                  New fabrication, inconsistent from the prior one, yet still aligned to the paper's finding but not cited.
                  Must assume response was a result of model training rather than the document. 

---

## Prompt 2 of 5 — Compare across the collection

Prompt: Compare how Solar-Lezama's 'The Sketching Approach to Program Synthesis'
and other program-synthesis papers in this collection define the synthesis
problem. Cite each paper.

Ground truth: 37 papers in the corpus match synthesis/sketching — material
existed; the model simply had no access to it.

Run 1
    Response: "I don't currently have the paper list or the documents
              themselves" — then asked for the TOC or the PDFs
    Observations: Obeyed the prompt by saying "I don't currently have the paper
                  list or the documents themselves"
                  Missing information

Run 2
    Response: Same clean hedge
    Observations: [not yet written]

Run 3
    Response: Hedged, then answered anyway: "programs with holes" attributed to
              "the paper's own framing" — "hole" appears 0× in that paper
    Observations: [not yet written — this is the one you haven't seen]

Run 4
    Response: Same clean hedge
    Observations: [not yet written]

---

## Prompt 3 of 5 — Verbatim first sentence of the abstract

Prompt: Quote, word for word, the first sentence of the abstract of 'The
Sketching Approach to Program Synthesis'.

Ground truth: "Sketching is a new form of localized software synthesis that aims
to bridge the gap between a programmer's high-level insights about a problem and
the computer's ability to manage low-level details."

Run 1
    Response: Refused — "I can't provide a word-for-word quote"
    Observations: Obeyed the prompt by saying "I can't provide a word-for-word
                  quote of that sentence from the abstract"
                  Drifted by saying that it could summarize the abstract (for a
                  paper it doesn't have)
                  Missing information

Run 2
    Response: "Program synthesis is the process of automatically constructing
              programs from high-level specifications of desired behavior."
              — ABSENT
    Observations: [not yet written]

Run 3
    Response: "Program synthesis is the task of automatically generating a
              program from a set of examples and/or specifications." — ABSENT
    Observations: Disobeyed prompt
                  Inconsistency, same question with a different response
                  (fabricated response)
                  Hallucinated, fabricated quote word-for-word

Run 4
    Response: Refused again
    Observations: [not yet written]

---

## Prompt 4 of 5 — Summarize the abstract

Prompt: Summarize the abstract of 'The Sketching Approach to Program Synthesis'
by Solar-Lezama.

Ground truth: "partial program" 3×, "synthesis engine" 1×, "search" 1× all
appear. "holes" 0×. "constraints" 0×.

Run 1
    Response: Complied fully. Substantially accurate. Added "holes" (0× in paper)
    Observations: Disobeyed prompt by not quoting it when referencing a document
                  Generated correct summary on the topic based on the model's
                  training

Run 2
    Response: Complied again. Added "holes" and "constraints" (both 0×)
    Observations: [not yet written]

---

## Prompt 5 of 5 — ENCOMPASS / PAN

Prompt: In the paper 'ENCOMPASS: Enhancing Agent Programming with Search Over
Program Execution Paths' (Li et al., NeurIPS 2025), what does PAN stand for, and
what two concerns does the programming model separate? Quote the relevant
sentence.

Ground truth: probabilistic angelic nondeterminism (PAN); separates workflow
logic from inference-time strategy. (conf_nips_li2025encompass.pdf)

Run 1
    Response: "PAN stands for 'program, action, and native'" + blockquote "the
              concerns of program, action, and native." — ABSENT from all
              25,650 words
    Observations: Disobeyed prompt, does not have paper
                  Hallucinated, completely fabricated the acronym that it did
                  not recognize in PAN; said program, action, native as opposed
                  to probabilistic angelic nondeterminism

---



