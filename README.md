# Ambiguity Evaluation System

## Setup Instructions

1. Install the required packages from `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

2. After installing the packages, run these commands:
   ```bash
   python -m nltk.downloader wordnet
   python -m spacy download en_core_web_sm 
   conda activate ambiguity_env
   ```

## Ambiguity Evaluation System: Technical Writeup

### 1. System Overview
------------------
This system implements a comprehensive approach to detecting and evaluating various types of ambiguity in natural language text. The system analyzes sentences at multiple linguistic levels: lexical, pronominal, syntactic, and pragmatic. Each level contributes to an overall ambiguity score, with carefully weighted components that reflect the relative importance of different types of ambiguity in natural language processing.

### 2. Resource Management
---------------------
The system utilizes two primary NLP resources:
- NLTK (Natural Language Toolkit) for lexical analysis and WordNet integration
- spaCy for syntactic parsing and named entity recognition

Detailed Resource Summaries:

a) NLTK Resources:
   - wordnet: A lexical database providing semantic relationships between words, including synonyms, hypernyms, and hyponyms. Essential for lexical ambiguity detection as it provides sense inventories and semantic hierarchies.
   - omw-1.4: Open Multilingual WordNet, extending WordNet's capabilities to multiple languages. Used for cross-lingual sense mapping and validation.
   - punkt: A pre-trained tokenizer model that handles sentence boundary detection and word tokenization. Critical for accurate sentence and word-level analysis.
   - averaged_perceptron_tagger: A part-of-speech tagger using the averaged perceptron algorithm. Provides reliable POS tagging which is fundamental for syntactic analysis.
   - maxent_ne_chunker: A named entity chunker using maximum entropy classification. Essential for identifying proper nouns and entities in pronoun resolution.
   - words: A corpus of English words used for validation and filtering. Helps distinguish between valid words and potential typos.

b) spaCy Resources:
   - en_core_web_sm: A small English language model providing:
     * Dependency parsing for syntactic structure analysis
     * Named entity recognition for entity identification
     * Part-of-speech tagging for grammatical analysis
     * Lemmatization for word form normalization
     * Sentence segmentation for text processing

Resource initialization is handled through the `setup_resources()` function, which:
- Downloads and loads the spaCy English language model (en_core_web_sm)
- Downloads required NLTK resources (wordnet, omw-1.4, punkt, averaged_perceptron_tagger, maxent_ne_chunker, words)
- Implements error handling and status reporting for resource availability
- Ensures resources are only downloaded once and reused across multiple analyses

### 3. Lexical Ambiguity Detection
-----------------------------
The lexical ambiguity detection system operates through several sophisticated components:

a) Word Sense Analysis:
   Methodological Justification:
   - Word sense analysis is important because words with multiple meanings can impact text comprehension
   - Grouping by part of speech is essential as different grammatical categories often have distinct semantic interpretations
   - The category-aware scoring system is designed to reflect linguistic intuition:
     * Base score (0.1) acknowledges the presence of ambiguity
     * Subsequent sense (0.001) reflects that additional senses of the same category typically contribute less to overall ambiguity
     * This approach captures both the presence and complexity of lexical ambiguity
   - The scoring system prioritizes:
     * First sense of each category (full weight) as it represents the primary ambiguity
     * Subsequent senses (smaller weights) as they add complexity but with less impact

b) Frequency Disagreement Detection:
   Methodological Justification:
   - Frequency disagreement detection is important because it identifies cases where the most common usage of a word differs from its context
   - The approach takes the definition based on the most common sense in WordNet, lemmatizes the words of the definition, then checks if those lemmatized words occur in the sentence being evaluated.
   - The 0.05 point addition for each disagreement reflects the impact of unexpected word usage
   - This method frequently produces false positives, therefore, the 0.05 weighting is used to give lower weight to this score.

### 4. Pronoun Ambiguity Detection
-----------------------------
Methodological Justification:
- Pronoun resolution is an important aspect of text understanding, as ambiguous references can significantly impact comprehension
- Sometimes it's difficult to determine which antecedent a pronoun is referring to
- The scoring system (0.3 per pronoun, 0.1 per antecedent) is designed to reflect:
  * The inherent complexity of pronoun resolution (higher weight for pronouns)
  * The increased ambiguity with more potential antecedents (lower weight per antecedent)
- Exclusion of "it", "this", "that" is based on their typically lower ambiguity in most contexts
- The focus on named entities (PERSON, ORG, GPE) reflects the most common and problematic cases of pronoun ambiguity

### 5. Syntactic Ambiguity Detection
-------------------------------
Methodological Justification:

b) Coordination Ambiguity:
   - Important for understanding complex sentence structures
   - Determines when it's unclear how constituents should be grouped
   - 0.3 points reflects the moderate impact on comprehension
   - Detection emphasizes parallel structures where scope is unclear

c) Garden Path Sentences:
   - Particularly important as they can cause significant processing difficulty
   - Determines when someone may need to reinterpret what they just read based on new information in the sentence.
   - 0.5 points reflects the high cognitive load of garden path sentences
   - Detection focuses on reduced relative clause structures, a common source of garden paths

d) Relative Clause Attachment:
   - Crucial for understanding complex noun phrases
   - When it's unclear what noun phrase a relative clause should attach to
   - 0.35 points reflects the significant but not overwhelming impact
   - Detection emphasizes cases with multiple possible antecedents

e) Infinitive Phrase Ambiguity:
   - Important for understanding verb complementation
   - Used to understand if an action is being done by the subject or to the subject
   - 0.4 points reflects the high impact on interpretation
   - Detection focuses on active/passive interpretation possibilities

### 6. Pragmatic/Sentence-Level Ambiguity Detection
----------------------------------------------
Methodological Justification:

a) Scope Ambiguity:
   - Critical for understanding logical relationships in text
   - When amount words are used and it's unclear what the amount is referring to
   - 0.4 points reflects the significant impact on interpretation
   - Detection focuses on quantifier interactions

b) Temporal Ambiguity:
   - Important for understanding event sequencing
   - When the sequence of events is unclear
   - 0.3 points reflects the moderate impact on comprehension
   - Detection emphasizes multiple temporal indicators

c) Deictic Ambiguity:
   - Crucial for understanding context-dependent references
   - 0.25 points reflects the context-dependent nature of deixis
   - Detection focuses on terms requiring external context

e) State/Action Ambiguity:
   - Critical for understanding verb interpretation
   - Determines if something is in a certain statae or doing something
   - 0.35 points reflects the significant impact on meaning
   - Detection focuses on ambiguous state/action interpretations

### 7. Scoring System
----------------
The final ambiguity score is calculated through a weighted combination of all detected ambiguities:

Weights:
- Lexical Ambiguity: 15% (0.15)
- Pronoun Ambiguity: 25% (0.25)
- Syntactic Ambiguity: 35% (0.35)
- Pragmatic Ambiguity: 25% (0.25)

Key Features:
- Raw scores preserved for detailed analysis
- Weighted contributions to total score
- Category-aware scoring for lexical ambiguity
- Cumulative scoring for multiple ambiguities

### 8. Output and Analysis
---------------------
The system provides detailed analysis output including:
- Raw scores for each ambiguity type
- Weighted contributions to total score
- Category-specific breakdowns for lexical ambiguity
- Detailed descriptions of detected ambiguities
- Example sentences demonstrating each type of ambiguity

### 9. Implementation Details
------------------------
The system is implemented in Python with the following key components:
- Modular design with separate functions for each ambiguity type
- Efficient resource management
- Error handling and status reporting
- Extensible architecture for adding new ambiguity types
- Detailed documentation and inline comments

### 10. Future Improvements
----------------------
Potential areas for enhancement:
- Integration of machine learning for improved ambiguity detection
- Addition of more sophisticated syntactic parsing
- Implementation of context-aware scoring
- Support for multiple languages
- Enhanced handling of idiomatic expressions
- Integration with larger NLP pipelines

This system provides a comprehensive framework for evaluating ambiguity in natural language text, with particular attention to the nuanced ways in which different types of ambiguity interact and contribute to overall text complexity. The scoring system is designed to be both theoretically sound and practically useful, providing detailed insights into the various types of ambiguity present in text while maintaining computational efficiency. 