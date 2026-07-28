# CBSE RAG — Chunk Metadata Reference

> Generated from `ingest.py` — documents every metadata field stored per vector chunk in ChromaDB, organised by subject group.  
> Re-ingest (`--force` or `--adhoc`) whenever this schema changes.

---

## Universal Fields

These fields are written for **every chunk across all subjects**.

| Field | Type | Example | Description |
|---|---|---|---|
| `source` | `str` | `"science_ch3.pdf"` | PDF filename the chunk came from |
| `class` | `str` | `"class_10"` | Class folder (`class_8`, `class_9`, `class_10`) |
| `subject` | `str` | `"science"` | Subject folder name (lowercase) |
| `page` | `int` | `12` | 0-based page number within the PDF |
| `chapter` | `str` | `"Chapter 3 – Metals and Non-Metals"` | Nearest chapter heading, carried forward across chunks |
| `chapter_num` | `int` | `3` | Numeric chapter number extracted from the heading |
| `section` | `str` | `"3.2 Physical Properties of Metals"` | Nearest section heading (`N.N Title`) |
| `subsection` | `str` | `"3.2.1 Malleability"` | Nearest subsection heading (`N.N.N Title`) |
| `content_type` | `str` | `"definition"` | Semantic content category — see per-subject tables below |
| `bloom_level` | `str` | `"apply"` | Estimated Bloom's taxonomy level for difficulty routing |
| `has_formula` | `bool` | `True` | Chunk contains a mathematical or chemical formula/unit |
| `has_table` | `bool` | `True` | Chunk appears to contain tabular data |
| `has_figure_ref` | `bool` | `True` | Chunk references a Figure / Diagram |
| `keyword_hints` | `str` | `"acid, base, salt, reaction, neutralisation"` | Top-5 content words (comma-separated) for debug/search |
| `word_count` | `int` | `87` | Word count of the chunk text |
| `lang_sub_type` | `str` | `""` | Language-only field — empty string for non-language subjects |
| `sentence_class` | `str` | `""` | Language-only field — empty string for non-language subjects |

---

## `content_type` Values (all subjects)

| Value | Detected when chunk contains… | Typical Bloom level |
|---|---|---|
| `question` | `"Q.1"`, `"1. What is…"`, `"explain"`, `"calculate"`, `"solve"` | apply |
| `answer` | `"Solution:"`, `"Ans:"`, `"Answer:"` | understand |
| `definition` | `"is defined as"`, `"is called"`, `"refers to"`, `"means that"` | remember |
| `fact` | Short declarative sentence (< 60 words) with `is/are/was/were` | remember |
| `summary` | `"What you have learnt"`, `"Key points"`, `"Points to remember"`, `"Recap"` | remember |
| `example` | `"Example 1:"`, `"Example:"` block | apply |
| `formula` | Algebraic equation (`v = u + at`), chemical formula (`H₂SO₄`), Greek symbols, units | apply |
| `table` | Markdown table pipes, tab-separated rows, `"S.No."` header | remember / understand |
| `figure_ref` | `"Fig. 1"`, `"as shown in"`, `"refer to diagram"` | understand |
| `activity` | `"Activity 1"`, `"Lab activity"`, `"Let us try"`, `"Experiment"` | apply |
| `note` | `"Note:"`, `"Remember:"`, `"Important:"`, `"Did you know?"` | remember |
| `exercise` | `"Exercises"`, `"Practice questions"`, `"NCERT Solutions"` (section header) | apply |
| `introduction` | `"Introduction"`, `"In this chapter we will"`, `"Let us begin"` | remember |
| `body` | General explanatory prose — fallback when no signal matches | remember |

---

## `bloom_level` Values

| Value | Bloom's Level | Triggered by |
|---|---|---|
| `remember` | L1 | `definition`, `fact`, `summary`, `introduction`, `body` types; no higher-order verbs found |
| `understand` | L2 | `"explain"`, `"describe"`, `"summarize"`, `"interpret"`, `"discuss"` |
| `apply` | L3 | `"calculate"`, `"solve"`, `"apply"`, `"demonstrate"`, `"construct"`, `"derive"` |
| `analyse` | L4 | `"analyse"`, `"differentiate"`, `"classify"`, `"why does"`, `"what would happen"` |
| `evaluate` | L5–L6 | `"evaluate"`, `"justify"`, `"critique"`, `"predict"`, `"design"`, `"hypothesize"` |

---

## Subject-wise Metadata Detail

---

### 📐 Mathematics

**Folder:** `data/class_*/mathematics/`

**What NCERT Maths PDFs contain and how they are tagged:**

| Content in PDF | `content_type` | `has_formula` | `bloom_level` | Notes |
|---|---|---|---|---|
| Theorem statement | `definition` | maybe | `remember` | "A triangle is defined as…" |
| Proof / derivation | `example` or `body` | `True` | `apply` / `analyse` | Step-by-step derivation |
| Worked example | `example` | `True` | `apply` | "Example 3: Find the value of x…" |
| Formula / equation | `formula` | `True` | `apply` | `v = u + at`, `A = πr²` |
| Exercise questions | `question` | maybe | `apply` / `analyse` | "1. Calculate…", "3. Prove that…" |
| Solutions (if present) | `answer` | `True` | `understand` | "Solution: Step 1…" |
| Summary / revision box | `summary` | maybe | `remember` | "Key concepts", "Points to remember" |
| Think & Discuss | `question` | `False` | `analyse` | Exploration questions in boxes |
| Activity / construction | `activity` | `False` | `apply` | Compass-and-ruler constructions |
| Data / frequency table | `table` | `False` | `understand` | Statistics chapters |
| Introduction paragraph | `introduction` | `False` | `remember` | Chapter openers |

**Key `has_formula` triggers for Maths:**
- Algebraic expressions: `x = (-b ± √(b²-4ac)) / 2a`
- Ratio / proportion: `a × d = b × c`
- Geometry: `Area = ½ × base × height`
- Statistics: `Mean = Σf·x / Σf`
- Units: `cm`, `m²`, `km/h`

**Worksheet generation priority:** `question` → `example` → `formula` → `answer` → `definition` → `summary`

---

### 🔬 Science (Classes 8–10 combined)

**Folder:** `data/class_*/science/`

Science covers Physics, Chemistry, and Biology content within the same PDF. The enrichment handles all three.

| Content in PDF | `content_type` | `has_formula` | `has_figure_ref` | `bloom_level` |
|---|---|---|---|---|
| Law / principle definition | `definition` | maybe | `False` | `remember` |
| Chemical equation | `formula` | `True` | `False` | `apply` |
| Reaction type description | `definition` or `body` | `True` | `False` | `understand` |
| Diagram description | `figure_ref` | `False` | `True` | `understand` |
| Lab experiment | `activity` | maybe | maybe | `apply` |
| Numerical problem | `question` | `True` | `False` | `apply` |
| Worked numerical | `example` | `True` | `False` | `apply` |
| Think & Discuss / In-text Q | `question` | `False` | `False` | `analyse` |
| End-of-chapter exercises | `question` / `exercise` | maybe | `False` | `apply`–`evaluate` |
| Summary / key takeaways | `summary` | `False` | `False` | `remember` |
| Caution / Note box | `note` | `False` | `False` | `remember` |
| Data / classification table | `table` | `False` | `False` | `understand` |
| Chapter introduction | `introduction` | `False` | `False` | `remember` |

**`has_formula` triggers for Science:**
- Chemical formulae: `H₂SO₄`, `CO₂`, `NaCl`, `Fe₂O₃`
- Physics equations: `F = ma`, `v = u + at`, `P = VI`
- Units: `mol`, `kg`, `kJ`, `kPa`, `atm`, `°C`, `Hz`, `N/m`
- Greek symbols: `α`, `β`, `γ`, `λ`, `Δ`, `Σ`

**`has_figure_ref` triggers:** `Fig. 3.1`, `diagram 2`, `as shown in`, `refer to figure`

**Worksheet generation priority:** `question` → `activity` → `example` → `formula` → `definition` → `fact` → `note` → `summary`

---

### 🌏 Social Science

**Folder:** `data/class_*/socialscience/`

Social Science covers History, Geography, Political Science, and Economics.

| Content in PDF | `content_type` | `has_table` | `bloom_level` | Notes |
|---|---|---|---|---|
| Event / term definition | `definition` | `False` | `remember` | "The French Revolution is…" |
| Historical fact | `fact` | `False` | `remember` | Short declarative sentences |
| Map / diagram reference | `figure_ref` | `False` | `understand` | "Refer to Map 3.1" |
| Data table (GDP, census) | `table` | `True` | `understand` / `analyse` | Economic / geographic data |
| Timeline / sequence | `body` | maybe | `understand` | Narrative chronology |
| End-of-chapter questions | `question` | `False` | `apply`–`evaluate` | "Why did…", "How did…", "Discuss" |
| Let's Recall / Revise | `summary` | `False` | `remember` | Review sections |
| Activity / project work | `activity` | `False` | `apply` | Map-marking, data collection |
| Important note | `note` | `False` | `remember` | "Note:", "Remember:" |
| Chapter introduction | `introduction` | `False` | `remember` | Scene-setting paragraphs |
| Worked example (Economics) | `example` | maybe | `apply` | Demand/supply calculations |
| Exercise section | `exercise` | `False` | `apply` | "Exercises", "Practice Questions" |

**Bloom level distribution for Social Science:**
- History chapters → mostly `remember` + `understand` (events, causes, effects)
- Geography chapters → `understand` + `analyse` (map reading, climate analysis)
- Economics chapters → `apply` + `analyse` (data interpretation, graphs)
- Political Science → `evaluate` (democratic values, rights debates)

**Worksheet generation priority:** `question` → `fact` → `definition` → `table` → `figure_ref` → `activity` → `summary`

---

### 🇬🇧 English

**Folder:** `data/class_*/english/`

English has the richest metadata of all subjects. Every chunk gets both universal fields **and** the two language-only fields.

#### Universal fields (same as all subjects)

| Content in PDF | `content_type` |
|---|---|
| Grammar rule paragraph | `definition` |
| Grammar exercise section | `exercise` |
| Comprehension questions | `question` |
| Worked grammar examples | `example` |
| Chapter summary | `summary` |
| Introduction to chapter | `introduction` |

#### Language-specific fields

| Field | Values for English |
|---|---|
| `lang_sub_type` | See full table below |
| `sentence_class` | `word_list` / `short_sentence` / `long_sentence` / `passage` |

#### `lang_sub_type` values for English

| `lang_sub_type` | Detected when chunk contains… | Worksheet use |
|---|---|---|
| `grammar_rule` | `"A noun is…"`, `"Present tense…"`, `"Active voice"`, `"Direct/Indirect speech"`, `"Subordinate clause"` | Grammar MCQ, rule-based fill-blank |
| `grammar_example` | `"e.g."`, `"For example:"`, `"Example: She runs fast."`, `"Correct/Incorrect:"` | Show-and-apply questions |
| `grammar_exercise` | `"Fill in the blanks"`, `"Rewrite the following"`, `"Do as directed"`, `"Match the columns"`, `"Underline the noun"` | Direct fill-blank, rewrite, MCQ |
| `comprehension` | `"Read the following passage and answer"`, `"Unseen passage"`, `"Based on the above extract"` | Passage + sub-questions |
| `comprehension_question` | `"Answer the following questions based on the passage"`, `"According to the passage"` | Q&A under a passage |
| `short_answer_q` | `"Answer briefly"`, `"In not more than 30 words"`, `"Name/Define/State"` | Short-answer questions |
| `long_answer_q` | `"Write in detail"`, `"In not less than 100 words"`, `"Write an essay/paragraph on"` | Essay / long-answer prompts |
| `vocabulary` | `"Word meanings"`, `"Synonyms/Antonyms"`, `"Homophones"`, `"Word bank"`, `"Difficult words"` | Synonym MCQ, match-the-word |
| `dialogue` | Two or more speaker-name lines (`"Ram: … Sita: …"`), `"Conversation between"` | Complete-the-dialogue questions |
| `letter_writing` | `"Formal/informal letter"`, `"Write a letter to"`, `"Yours sincerely"`, `"Dear Sir/Madam"` | Letter-writing prompts |
| `essay_writing` | `"Write an essay on"`, `"Composition on"` | Essay-writing prompts |
| `story` | `"Once upon a time"`, `"Story of"`, `"Write a story"` | Story comprehension or narrative writing |
| `poem` | `"poem"`, `"poetry"`, `"stanza"`, `"rhyme"`, `"verse"`, `"couplet"` | Poem appreciation, stanza questions |
| `translation` | `"Translate the following"`, `"Translation into"` | Translation exercises |
| `note_making` | `"Note-making"`, `"Make notes from the passage"`, `"Summary writing"` | Note-making questions |
| `report_writing` | `"Report writing"`, `"Write a news report"`, `"Newspaper report"` | Report-writing prompts |
| `speech` | `"Speech writing"`, `"Write a speech on"`, `"Debate on"` | Speech/debate prompts |
| `body` | General prose, narrative body text | Background context for questions |

#### `sentence_class` values for English

| Value | Condition | Best matched question types |
|---|---|---|
| `word_list` | Avg < 4 words/line, total < 60 words | Vocabulary MCQ, match-the-word, fill-blank |
| `short_sentence` | Avg < 12 words/line, total < 120 words | Grammar drills, fill-in-blank, True/False |
| `long_sentence` | Avg 12–25 words/line | Short-answer questions, grammar analysis |
| `passage` | Total ≥ 200 words | Comprehension, essay questions, note-making |

---

### 🇮🇳 Hindi

**Folder:** `data/class_*/hindi/`

Hindi has the same field structure as English. Patterns detect **Devanagari script** keywords.

#### `lang_sub_type` values for Hindi

| `lang_sub_type` | Hindi signal words / phrases detected |
|---|---|
| `grammar_rule` | `कारक`, `संधि`, `समास`, `वचन`, `लिंग`, `काल`, `विभक्ति`, `क्रिया`, `विशेषण`, `सर्वनाम` |
| `grammar_example` | `उदाहरण`, `उदाहरण :`, `उदा.` |
| `grammar_exercise` | `रिक्त स्थान`, `सही शब्द भरिए`, `वाक्य बनाइए` |
| `comprehension` | `गद्यांश`, `पद्यांश`, `अपठित गद्यांश` |
| `comprehension_question` | `निम्नलिखित प्रश्नों के उत्तर दीजिए` |
| `short_answer_q` | `संक्षेप में`, `एक शब्द में`, `एक वाक्य में` |
| `long_answer_q` | `विस्तार से लिखिए`, `निबंध लिखिए` |
| `dialogue` | `वार्तालाप`, `संवाद` |
| `letter_writing` | `पत्र लेखन`, `औपचारिक पत्र`, `अनौपचारिक पत्र` |
| `essay_writing` | `निबंध लेखन`, `निबंध लिखिए`, `निबंध लिखो` |
| `story` | `कहानी लेखन`, `कहानी लिखिए`, `एक बार की बात` |
| `poem` | `कविता`, `पद्य`, `दोहा`, `चौपाई`, `श्लोक` |
| `vocabulary` | `शब्दार्थ`, `पर्यायवाची`, `विलोम शब्द`, `मुहावरे`, `लोकोक्तियाँ` |
| `translation` | `अनुवाद कीजिए`, `अनुवाद करो`, `अनुवाद लिखिए` |
| `note_making` | English patterns (note-making, summary writing) |
| `report_writing` | English patterns (report writing, newspaper report) |
| `speech` | English patterns (speech writing, debate) |
| `body` | General Hindi prose |

**`sentence_class`** — same logic as English (word count thresholds), applied to Devanagari text.

---

### 🌸 Kannada

**Folder:** `data/class_*/kannada/`

#### `lang_sub_type` values for Kannada

| `lang_sub_type` | Kannada signal words detected |
|---|---|
| `grammar_rule` | `ಸಂಧಿ`, `ಸಮಾಸ`, `ಕಾರಕ`, `ಕ್ರಿಯಾ`, `ವಿಭಕ್ತಿ`, `ನಾಮಪದ`, `ಕ್ರಿಯಾಪದ` |
| `grammar_example` | `ಉದಾಹರಣೆ`, `ಉದಾ.` |
| `grammar_exercise` | `ಖಾಲಿ ತುಂಬಿರಿ`, `ವಾಕ್ಯ ರಚಿಸಿ` |
| `comprehension` | `ಗद्यभाग`, `ಪद्यभाग`, `ಅಪಠಿತ ಗद्यभाग` |
| `comprehension_question` | `ಕೆಳಗಿನ ಪ್ರಶ್ನೆಗಳಿಗೆ ಉತ್ತರಿಸಿ` |
| `short_answer_q` | `ಸಂಕ್ಷಿಪ್ತವಾಗಿ`, `ಒಂದು ವಾಕ್ಯದಲ್ಲಿ` |
| `long_answer_q` | `ವಿವರವಾಗಿ ಬರೆಯಿರಿ`, `ಪ್ರಬಂಧ ಬರೆಯಿರಿ` |
| `dialogue` | `ಸಂಭಾಷಣೆ` |
| `letter_writing` | `ಪತ್ರ ಬರೆಯಿರಿ`, `ಔಪಚಾರಿಕ ಪತ್ರ` |
| `essay_writing` | `ಪ್ರಬಂಧ ರಚಿಸಿ` |
| `story` | `ಕಥೆ ಬರೆಯಿರಿ`, `ಒಮ್ಮೆ ಒಬ್ಬ` |
| `poem` | `ಕವಿತೆ`, `ಪದ್ಯ` |
| `vocabulary` | `ಶಬ್ದಾರ್ಥ`, `ಸಮಾನಾರ್ಥಕ`, `ವಿರುದ್ಧಾರ್ಥಕ` |
| `translation` | `ಅನುವಾದ ಮಾಡಿ`, `ಭಾಷಾಂತರ ಮಾಡಿ` |
| `note_making` / `report_writing` / `speech` | English patterns |
| `body` | General Kannada prose |

---

### 🌺 Tamil

**Folder:** `data/class_*/tamil/`

#### `lang_sub_type` values for Tamil

| `lang_sub_type` | Tamil signal words detected |
|---|---|
| `grammar_rule` | `சந்தி`, `வேர்ச்சொல்`, `வினையெச்சம்`, `பெயரெச்சம்`, `விகுதி` |
| `grammar_example` | `எடுத்துக்காட்டு`, `எ.கா.` |
| `grammar_exercise` | `வெற்றிட நிரப்புக`, `சொற்றொடர் அமை` |
| `comprehension` | `உரைநடை பகுதி`, `கவிதை பகுதி` |
| `dialogue` | `உரையாடல்` |
| `vocabulary` | `சொற்பொருள்`, `எதிர்ச்சொல்`, `ஒத்த சொல்` |
| `translation` | `மொழிபெயர்`, `மொழிபெயர்ப்பு` |
| `poem` | `கவிதை`, `பாடல்` |
| All others | English patterns (short/long answer, essay, letter, story, etc.) |
| `body` | General Tamil prose |

---

### 🕉️ Sanskrit

**Folder:** `data/class_*/sanskrit/`

#### `lang_sub_type` values for Sanskrit

| `lang_sub_type` | Sanskrit signal words detected |
|---|---|
| `grammar_rule` | `संधि`, `समास`, `कारक`, `विभक्ति`, `धातु`, `प्रत्यय`, `उपसर्ग` |
| `grammar_example` | `उदाहरण`, `यथा` |
| `poem` | `श्लोक`, `पद्य` |
| `translation` | `अनुवाद कीजिए`, `अनुवाद करो` |
| All others | English patterns (fill-blank, rewrite, comprehension, etc.) |
| `body` | General Sanskrit prose / shloka text |

---

## How All Fields Work Together

### Retrieval scoring formula (in `api.py`)

```
final_score = confidence             # CrossEncoder similarity (0-100)
            + bloom_boost            # +15 if bloom_level matches difficulty
            + type_boost             # content_type or lang_sub_type rank × 1.5–2.0
            + sentence_class_boost   # +8 if sentence_class suits question type
            + formula_boost          # +3 for Maths/Science when has_formula=True
```

### Sentence class → question type mapping

| Question type requested | Preferred `sentence_class` |
|---|---|
| `fillblank` | `short_sentence`, `word_list` |
| `short` | `short_sentence`, `long_sentence` |
| `long` | `passage`, `long_sentence` |
| `mcq` | `short_sentence`, `long_sentence`, `passage` |
| `truefalse` | `short_sentence`, `long_sentence` |

### Context prefix format in ChromaDB

Each chunk's embedded text starts with a structured prefix so vector similarity naturally clusters chunks by location and type:

```
Chapter 3 – Metals and Non-Metals | 3.2 Physical Properties | 3.2.1 Malleability | [definition] | [bloom:remember]
<chunk body text>
```

For language subjects:
```
Chapter 4 – The Last Lesson | 4.1 Reading Section | [comprehension] | [comprehension] | [passage] | [bloom:understand]
<chunk body text>
```

---

## Full Field Reference (compact)

| Field | Subjects | Type | Possible values |
|---|---|---|---|
| `source` | All | `str` | PDF filename |
| `class` | All | `str` | `class_8`, `class_9`, `class_10` |
| `subject` | All | `str` | `mathematics`, `science`, `socialscience`, `english`, `hindi`, `kannada`, `tamil`, `sanskrit` |
| `page` | All | `int` | `0`–`N` |
| `chapter` | All | `str` | `"Chapter N – Title"` or `""` |
| `chapter_num` | All | `int` | `0`–`15` (0 = not found) |
| `section` | All | `str` | `"N.N Title"` or `""` |
| `subsection` | All | `str` | `"N.N.N Title"` or `""` |
| `content_type` | All | `str` | `question`, `answer`, `definition`, `fact`, `summary`, `example`, `formula`, `table`, `figure_ref`, `activity`, `note`, `exercise`, `introduction`, `body` |
| `bloom_level` | All | `str` | `remember`, `understand`, `apply`, `analyse`, `evaluate` |
| `has_formula` | All | `bool` | `True` / `False` |
| `has_table` | All | `bool` | `True` / `False` |
| `has_figure_ref` | All | `bool` | `True` / `False` |
| `keyword_hints` | All | `str` | `"word1, word2, word3, word4, word5"` |
| `word_count` | All | `int` | `10`–`600` |
| `lang_sub_type` | Language only | `str` | `grammar_rule`, `grammar_example`, `grammar_exercise`, `comprehension`, `comprehension_question`, `short_answer_q`, `long_answer_q`, `dialogue`, `letter_writing`, `essay_writing`, `story`, `poem`, `summary_passage`, `vocabulary`, `translation`, `note_making`, `report_writing`, `speech`, `body`, `""` |
| `sentence_class` | Language only | `str` | `word_list`, `short_sentence`, `long_sentence`, `passage`, `""` |
