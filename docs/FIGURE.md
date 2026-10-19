# Overview figure

`assets/overview.png` was generated with the built-in image generation tool. It is a conceptual illustration of the research design, rather than an experimental result.

Generation prompt: Create a wide academic overview for CIVIC-CR on a white background, with hand-drawn black outlines and pastel mint, blue, lavender, and yellow panels. Show program and frozen reviewer producing a critique, three branches for target repair, unrelated repair, and behavior-preserving refactoring, frozen encoders, a trainable verifier, and the expected validity response for each branch. Target repair decreases validity; unrelated repair preserves validity; refactoring preserves validity while moving evidence. End with surface/reject. Use clear arrows and readable labels. Include no benchmark scores, conference logos, or invented results.

Final revision prompt: Preserve the layout and style. Use the genuine empty-list defect `def first(items): return items[0]` and critique “Empty input raises IndexError.” Show a target guard returning `None`, an unrelated independent resource-leak repair, and a behavior-preserving parameter rename to `values`. Remove the initial duplicate-counting example and code comments. Request program → reviewer → critique flow. The three verifier drawings denote shared weights; refactoring requires a corresponding critique/evidence mapping.
