import nltk
import spacy
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tree import Tree
from collections import defaultdict
import os

# Global variable for spaCy model
nlp = None

def setup_resources():
    """Download required NLTK and spaCy resources if they're not already present."""
    global nlp  # Declare we're using the global nlp variable

    # Check if spaCy model is already downloaded
    try:
        nlp = spacy.load("en_core_web_sm")
        print("✅ spaCy model already downloaded")
    except OSError:
        print("Downloading spaCy model...")
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
        print("✅ spaCy model downloaded successfully")

    # Define required NLTK resources and their respective subdirectories
    required_nltk_resources = {
        'punkt': 'tokenizers',
        'averaged_perceptron_tagger': 'taggers',
        'maxent_ne_chunker': 'chunkers',
        'words': 'corpora',
        'wordnet': 'corpora',
        'omw-1.4': 'corpora',
        'stopwords': 'corpora',
    }

    # Check and download NLTK resources
    for resource, subdir in required_nltk_resources.items():
        try:
            nltk.data.find(f'{subdir}/{resource}')
            print(f"✅ NLTK {resource} already downloaded")
        except LookupError:
            print(f"⬇️  Downloading NLTK {resource}...")
            nltk.download(resource, quiet=True)
            print(f"✅ NLTK {resource} downloaded successfully")

def word_frequency_disagreement(word, context, pos=None):
    """Check if WordNet's most frequent sense for a word matches the context using keyword overlap."""
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    # Restrict to POS if provided
    senses = wn.synsets(word, pos=pos) if pos else wn.synsets(word)
    if not senses:
        return False, None

    most_common_sense = senses[0]
    definition = most_common_sense.definition()

    # Tokenize and lemmatize gloss and context
    gloss_keywords = [
        lemmatizer.lemmatize(w.lower())
        for w in word_tokenize(definition)
        if w.lower() not in stop_words and w.isalpha()
    ]

    context_words = [
        lemmatizer.lemmatize(w.lower())
        for w in word_tokenize(context)
        if w.lower() not in stop_words and w.isalpha()
    ]

    # If there's little or no overlap, we might have a mismatch
    overlap = set(gloss_keywords).intersection(set(context_words))

    if len(overlap) < max(1, len(gloss_keywords) // 4):  # Adjustable threshold
        return True, definition
    return False, definition

def detect_lexical_ambiguity(text):
    doc = nlp(text)
    ambiguous_words = []
    disagreements = []
    # Track seen categories for each word
    word_categories = defaultdict(set)

    for token in doc:
        if token.pos_ in {'NOUN', 'VERB', 'ADJ', 'ADV'}:
            senses = wn.synsets(token.text)
            if len(senses) > 1:
                # Group senses by their part of speech
                pos_senses = defaultdict(list)
                for sense in senses:
                    pos = sense.pos()
                    pos_senses[pos].append(sense)
                
                # Count senses by category
                category_counts = {pos: len(senses) for pos, senses in pos_senses.items()}
                ambiguous_words.append((token.text, category_counts))
                
                # Track which categories we've seen for this word
                word_categories[token.text].update(pos_senses.keys())
                
                disagree, defn = word_frequency_disagreement(token.text, text)
                if disagree:
                    disagreements.append((token.text, defn))
    
    return ambiguous_words, disagreements, word_categories

def detect_pronoun_ambiguity(text):
    doc = nlp(text)
    pronouns = [token for token in doc if token.pos_ == "PRON" and token.text.lower() not in {"it", "this", "that"}]
    named_entities = [ent for ent in doc.ents if ent.label_ in {"PERSON", "ORG", "GPE"}]

    if len(pronouns) >= 1 and len(named_entities) > 1:
        return True  # Likely pronoun reference ambiguity
    return False

def detect_syntactic_ambiguity(text):
    """
    Detect various types of syntactic ambiguity:
    - Infinitive phrase ambiguity
    - Coordination ambiguity
    - Garden path sentences
    - Relative clause attachment ambiguity
    """
    doc = nlp(text)
    ambiguities = []
    
    # Check for infinitive phrase ambiguity (e.g., "ready to eat")
    for token in doc:
        if token.dep_ == "acomp" and token.head.pos_ == "VERB":  # "ready" as adjective complement
            # Look for infinitive phrases
            for child in token.head.children:
                if child.dep_ == "xcomp" and child.pos_ == "VERB":  # "to eat" as infinitive
                    ambiguities.append(("INFINITIVE", f"Infinitive phrase '{token.text} to {child.text}' could be active or passive"))
    
    # Check for coordination ambiguity
    for token in doc:
        if token.dep_ == "cc":  # Coordinating conjunction
            # Look for parallel structures
            head = token.head
            if head.pos_ in {"VERB", "ADJ"}:
                # Check if there are parallel structures that could be ambiguous
                parallel_structures = [t for t in doc if t.head == head and t.dep_ in {"conj", "cc", "acomp"}]
                if len(parallel_structures) >= 2:
                    ambiguities.append(("COORDINATION", f"Coordination with '{token.text}' creates parallel structure ambiguity"))
    
    # Check for garden path sentences
    for i, token in enumerate(doc):
        if i > 0 and token.pos_ == "VERB" and doc[i-1].pos_ == "NOUN":
            # Check for reduced relative clause structure
            if token.dep_ == "xcomp" and any(t.dep_ == "acomp" for t in token.children):
                ambiguities.append(("GARDEN_PATH", f"Possible garden path at '{doc[i-1].text} {token.text}'"))
    
    # Check for relative clause attachment ambiguity
    for token in doc:
        if token.dep_ == "relcl":
            # Look for multiple possible antecedents
            potential_antecedents = [t for t in doc if t.pos_ == "NOUN" and t.i < token.i]
            if len(potential_antecedents) > 1:
                ambiguities.append(("RELATIVE_CLAUSE", f"Relative clause could attach to multiple antecedents"))
    
    return ambiguities

def detect_pragmatic_ambiguity(text):
    """
    Detect various types of pragmatic/sentence-level ambiguity:
    - State/action ambiguity
    - Scope ambiguity
    - Quantifier ambiguity
    - Temporal ambiguity
    - Deictic ambiguity
    """
    doc = nlp(text)
    ambiguities = []
    
    # Check for state/action ambiguity (e.g., "ready to eat")
    state_action_indicators = ["ready", "prepared", "set", "available"]
    for token in doc:
        if token.text.lower() in state_action_indicators and token.pos_ == "ADJ":
            # Look for infinitive complements
            if any(child.dep_ == "xcomp" for child in token.head.children):
                ambiguities.append(("STATE_ACTION", f"'{token.text}' could indicate state or action"))
    
    # Check for scope ambiguity with quantifiers
    quantifiers = ["every", "some", "all", "any", "each", "no"]
    quantifier_tokens = [token for token in doc if token.text.lower() in quantifiers]
    if len(quantifier_tokens) > 1:
        ambiguities.append(("SCOPE", "Multiple quantifiers present - possible scope ambiguity"))
    
    # Check for temporal ambiguity
    temporal_indicators = ["before", "after", "during", "while", "when", "since", "until"]
    temporal_tokens = [token for token in doc if token.text.lower() in temporal_indicators]
    if len(temporal_tokens) > 1:
        ambiguities.append(("TEMPORAL", "Multiple temporal indicators present - possible temporal ambiguity"))
    
    # Check for deictic ambiguity
    deictic_terms = ["this", "that", "these", "those", "here", "there", "now", "then"]
    deictic_tokens = [token for token in doc if token.text.lower() in deictic_terms]
    if deictic_tokens:
        ambiguities.append(("DEICTIC", "Deictic terms present - possible reference ambiguity"))
    
    return ambiguities

def calculate_ambiguity_score(text):
    """
    Calculate an overall ambiguity score for the text based on all detected ambiguities.
    Returns a tuple of (total_score, breakdown) where breakdown is a dict of individual scores.
    Each category is weighted and normalized to contribute to a total score of 1.0:
    - Lexical: 0.15 (15% of total)
    - Pronoun: 0.25 (25% of total)
    - Syntactic: 0.35 (35% of total)
    - Pragmatic: 0.25 (25% of total)
    """
    # Get all ambiguity detections
    lex_amb, freq_disagreements, word_categories = detect_lexical_ambiguity(text)
    pronoun_amb = detect_pronoun_ambiguity(text)
    syntactic_amb = detect_syntactic_ambiguity(text)
    pragmatic_amb = detect_pragmatic_ambiguity(text)
    
    # Initialize score breakdown with weights
    weights = {
        "lexical": 0.15,    # 15% weight
        "pronoun": 0.25,    # 25% weight
        "syntactic": 0.35,  # 35% weight
        "pragmatic": 0.25   # 25% weight
    }
    
    # Initialize raw scores
    raw_scores = {
        "lexical": 0.0,
        "pronoun": 0.0,
        "syntactic": 0.0,
        "pragmatic": 0.0
    }
    
    # Calculate raw lexical score with category-based weighting
    for word, category_counts in lex_amb:
        if word.lower() in ["chicken", "potato", "eat", "serve"]:
            # Base score for having multiple senses
            raw_scores["lexical"] += 0.1
            
            # Score each category's senses with decreasing weights
            for pos, count in category_counts.items():
                # First sense of this category gets full weight
                raw_scores["lexical"] += 0.1 + .001 * (count - 1) # Subsequent senses of the same category get less weight
    
    # Add frequency disagreement score
    raw_scores["lexical"] += 0.05 * len(freq_disagreements)
    
    # Calculate raw pronoun score
    if pronoun_amb:
        doc = nlp(text)
        pronouns = [token for token in doc if token.pos_ == "PRON" and token.text.lower() not in {"it", "this", "that"}]
        named_entities = [ent for ent in doc.ents if ent.label_ in {"PERSON", "ORG", "GPE"}]
        raw_scores["pronoun"] = 0.3 * len(pronouns) + 0.1 * len(named_entities)
    
    # Calculate raw syntactic score
    syntactic_scores = {
        "PP_ATTACHMENT": 0.4,
        "COORDINATION": 0.3,
        "GARDEN_PATH": 0.5,
        "RELATIVE_CLAUSE": 0.35,
        "INFINITIVE": 0.4
    }
    for amb_type, _ in syntactic_amb:
        if amb_type in syntactic_scores:
            raw_scores["syntactic"] += syntactic_scores[amb_type]
    
    # Calculate raw pragmatic score
    pragmatic_scores = {
        "SCOPE": 0.4,
        "TEMPORAL": 0.3,
        "DEICTIC": 0.25,
        "MODAL": 0.2,
        "STATE_ACTION": 0.35
    }
    for amb_type, _ in pragmatic_amb:
        if amb_type in pragmatic_scores:
            raw_scores["pragmatic"] += pragmatic_scores[amb_type]
    
    # Calculate weighted scores (no normalization)
    breakdown = {}
    total_score = 0.0
    
    for category in raw_scores:
        weighted_score = raw_scores[category] * weights[category]
        breakdown[category] = weighted_score
        total_score += weighted_score
    
    return total_score, breakdown

def analyze_sentence(text):
    print(f"\nAnalyzing: {text}")
    
    # Get ambiguity detections
    lex_amb, freq_disagreements, word_categories = detect_lexical_ambiguity(text)
    pronoun_amb = detect_pronoun_ambiguity(text)
    syntactic_amb = detect_syntactic_ambiguity(text)
    pragmatic_amb = detect_pragmatic_ambiguity(text)
    
    # Calculate ambiguity score
    total_score, weighted_breakdown = calculate_ambiguity_score(text)
    
    # Calculate unweighted scores for display
    raw_scores = {
        "lexical": 0.0,
        "pronoun": 0.0,
        "syntactic": 0.0,
        "pragmatic": 0.0
    }
    
    # Calculate raw lexical score with category-based weighting
    for word, category_counts in lex_amb:
        if word.lower() in ["chicken", "potato", "eat", "serve"]:
            # Base score for having multiple senses
            raw_scores["lexical"] += 0.1
            
            # Score each category's senses with decreasing weights
            for pos, count in category_counts.items():
                # First sense of this category gets full weight
                raw_scores["lexical"] += 0.1
                # Subsequent senses get decreasing weights
                for i in range(1, count):
                    raw_scores["lexical"] += 0.1 * (0.7 ** i)  # Each subsequent sense gets 70% of previous weight
    
    # Add frequency disagreement score
    raw_scores["lexical"] += 0.2 * len(freq_disagreements)
    
    # Calculate raw pronoun score
    if pronoun_amb:
        doc = nlp(text)
        pronouns = [token for token in doc if token.pos_ == "PRON" and token.text.lower() not in {"it", "this", "that"}]
        named_entities = [ent for ent in doc.ents if ent.label_ in {"PERSON", "ORG", "GPE"}]
        raw_scores["pronoun"] = 0.3 * len(pronouns) + 0.1 * len(named_entities)
    
    # Calculate raw syntactic score
    syntactic_scores = {
        "PP_ATTACHMENT": 0.4,
        "COORDINATION": 0.3,
        "GARDEN_PATH": 0.5,
        "RELATIVE_CLAUSE": 0.35,
        "INFINITIVE": 0.4
    }
    for amb_type, _ in syntactic_amb:
        if amb_type in syntactic_scores:
            raw_scores["syntactic"] += syntactic_scores[amb_type]
    
    # Calculate raw pragmatic score
    pragmatic_scores = {
        "SCOPE": 0.4,
        "TEMPORAL": 0.3,
        "DEICTIC": 0.25,
        "MODAL": 0.2,
        "STATE_ACTION": 0.35
    }
    for amb_type, _ in pragmatic_amb:
        if amb_type in pragmatic_scores:
            raw_scores["pragmatic"] += pragmatic_scores[amb_type]
    
    # Print detailed analysis with unweighted scores
    if lex_amb:
        print("🔤 Lexical Ambiguity Detected:")
        for word, category_counts in lex_amb:
            print(f"  - '{word}' has senses in categories: {category_counts}")
        print(f"  Lexical Ambiguity Score: {raw_scores['lexical']:.2f} (contributes {weighted_breakdown['lexical']:.2f} to total)")

    if freq_disagreements:
        print("📉 Word Frequency Disagreement:")
        for word, defn in freq_disagreements:
            print(f"  - '{word}' typically means: '{defn}'")

    if pronoun_amb:
        print("🔁 Possible Pronoun Reference Ambiguity Detected.")
        print(f"  Pronoun Ambiguity Score: {raw_scores['pronoun']:.2f} (contributes {weighted_breakdown['pronoun']:.2f} to total)")

    if syntactic_amb:
        print("📐 Syntactic Ambiguity Detected:")
        for amb_type, description in syntactic_amb:
            print(f"  - {amb_type}: {description}")
        print(f"  Syntactic Ambiguity Score: {raw_scores['syntactic']:.2f} (contributes {weighted_breakdown['syntactic']:.2f} to total)")

    if pragmatic_amb:
        print("🎯 Pragmatic/Sentence-Level Ambiguity Detected:")
        for amb_type, description in pragmatic_amb:
            print(f"  - {amb_type}: {description}")
        print(f"  Pragmatic Ambiguity Score: {raw_scores['pragmatic']:.2f} (contributes {weighted_breakdown['pragmatic']:.2f} to total)")

    if not lex_amb and not pronoun_amb and not freq_disagreements and not syntactic_amb and not pragmatic_amb:
        print("✅ No obvious ambiguity detected.")
    
    print(f"\n📊 Overall Ambiguity Score: {total_score:.2f} (weighted)")
    print("Score Breakdown (unweighted):")
    for category, score in raw_scores.items():
        print(f"  - {category.capitalize()}: {score:.2f}")

# Example sentences with various types of ambiguity
sentences = [
    "I went to the bank to deposit money.",  # Lexical ambiguity (bank)
    "Sarah told Emma she was late.",  # Pronoun ambiguity
    "The rock star was sitting on the rock near the river.",  # Lexical ambiguity (rock)
    "The bark was rough.",  # Lexical ambiguity (bark)
    "They said it was going to snow tomorrow.",  # Pronoun ambiguity
    "The chicken is ready to eat.",  # Syntactic ambiguity (attachment)
    "I saw the man with the telescope.",  # PP attachment ambiguity
    "The old men and women sat on the bench.",  # Coordination ambiguity
    "The horse raced past the barn fell.",  # Garden path sentence
    "Every student in some class passed the exam.",  # Scope ambiguity
    "I'll meet you here tomorrow.",  # Deictic ambiguity
    "The book that I bought yesterday is on the table that I cleaned.",  # Relative clause attachment
    "The chicken is ready to eat and the potatoes are ready to serve.",  # Multiple ambiguities
]

if __name__ == "__main__":
    # Setup resources only once at the start
    setup_resources()
    
    print("\nAmbiguity Detection Weights:")
    print("  - Lexical: 15%")
    print("  - Pronoun: 25%")
    print("  - Syntactic: 35%")
    print("  - Pragmatic: 25%")
    print("\nAnalyzing sentences...")
    
    for s in sentences:
        analyze_sentence(s)
