"""
knowledge/curated_topics.py
────────────────────────────
Hardcoded GeeksforGeeks URLs for every ACD topic.
Used by gfg_scraper.py to pre-populate the knowledge base.
"""

# Each entry: (topic_label, URL)
GFG_ACD_URLS: list[tuple[str, str]] = [
    # ── Unit I ──────────────────────────────────────────────────────────────
    ("Introduction to Automata Theory", "https://www.geeksforgeeks.org/introduction-of-theory-of-computation/"),
    ("Finite Automata", "https://www.geeksforgeeks.org/introduction-of-finite-automata/"),
    ("DFA – Deterministic Finite Automaton", "https://www.geeksforgeeks.org/designing-deterministic-finite-automata-set-1/"),
    ("NFA – Non-Deterministic Finite Automaton", "https://www.geeksforgeeks.org/introduction-of-finite-automata/"),
    ("NFA to DFA Conversion", "https://www.geeksforgeeks.org/conversion-from-nfa-to-dfa/"),
    ("Regular Expressions", "https://www.geeksforgeeks.org/regular-expressions-in-automata-theory/"),
    ("RE to NFA (Thompson's Construction)", "https://www.geeksforgeeks.org/regular-expression-to-nfa/"),
    ("DFA Minimization", "https://www.geeksforgeeks.org/minimization-of-dfa/"),
    ("Pumping Lemma for Regular Languages", "https://www.geeksforgeeks.org/pumping-lemma/"),
    ("Lex – Lexical Analyzer Generator", "https://www.geeksforgeeks.org/flex-fast-lexical-analyzer-generator/"),

    # ── Unit II ─────────────────────────────────────────────────────────────
    ("Phases of Compiler", "https://www.geeksforgeeks.org/phases-of-a-compiler/"),
    ("Lexical Analysis", "https://www.geeksforgeeks.org/introduction-of-lexical-analysis/"),
    ("Context Free Grammar", "https://www.geeksforgeeks.org/classification-of-context-free-grammars/"),
    ("Parse Tree", "https://www.geeksforgeeks.org/parse-tree-in-compiler-design/"),
    ("Ambiguous Grammar", "https://www.geeksforgeeks.org/ambiguous-grammar/"),
    ("First and Follow Sets", "https://www.geeksforgeeks.org/first-set-in-syntax-analysis/"),
    ("LL(1) Parsing", "https://www.geeksforgeeks.org/construction-of-ll1-parsing-table/"),
    ("Bottom-Up Parsing", "https://www.geeksforgeeks.org/bottom-up-or-shift-reduce-parsers-in-compiler-design/"),
    ("LR Parsing", "https://www.geeksforgeeks.org/lr-parsing-set-1-canonical-sets-of-lr0-items/"),
    ("LALR Parser", "https://www.geeksforgeeks.org/lalr-parser-with-examples/"),
    ("YACC", "https://www.geeksforgeeks.org/introduction-to-yacc/"),

    # ── Unit III ────────────────────────────────────────────────────────────
    ("Syntax Directed Translation", "https://www.geeksforgeeks.org/syntax-directed-translation/"),
    ("S-Attributed and L-Attributed Grammars", "https://www.geeksforgeeks.org/s-attributed-and-l-attributed-grammars/"),
    ("Intermediate Code Generation", "https://www.geeksforgeeks.org/intermediate-code-generation-in-compiler-design/"),
    ("Three Address Code", "https://www.geeksforgeeks.org/three-address-code/"),
    ("Abstract Syntax Tree (AST)", "https://www.geeksforgeeks.org/abstract-syntax-tree-ast-in-java/"),
    ("Translation of Expressions", "https://www.geeksforgeeks.org/syntax-directed-translation-of-assignment-statement/"),
    ("Translation of Control Flow", "https://www.geeksforgeeks.org/code-generation-in-compiler-design/"),

    # ── Unit IV ─────────────────────────────────────────────────────────────
    ("Runtime Environments", "https://www.geeksforgeeks.org/runtime-environments-in-compiler-design/"),
    ("Storage Organization", "https://www.geeksforgeeks.org/storage-organization-in-compiler/"),
    ("Code Optimization", "https://www.geeksforgeeks.org/code-optimization-in-compiler-design/"),
    ("Peephole Optimization", "https://www.geeksforgeeks.org/peephole-optimization-in-compiler-design/"),
    ("Control Flow Graphs", "https://www.geeksforgeeks.org/control-flow-graph-in-compiler-design/"),
    ("Basic Blocks and Flow Graphs", "https://www.geeksforgeeks.org/basic-blocks-in-compiler-design/"),

    # ── Unit V ──────────────────────────────────────────────────────────────
    ("Code Generation", "https://www.geeksforgeeks.org/code-generation-in-compiler-design/"),
    ("Register Allocation", "https://www.geeksforgeeks.org/register-allocation-in-compiler-design/"),
    ("DAG Representation of Basic Blocks", "https://www.geeksforgeeks.org/dag-representation-of-basic-blocks/"),
]
