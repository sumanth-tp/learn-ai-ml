from pathlib import Path

from board import Board


OUT = Path(__file__).resolve().parents[2] / "static" / "img" / "ir"


def ir_pipeline():
    board = Board(1120, 400, "From documents to useful results", "Index once; interpret and rank each query")
    boxes = [
        board.card(30, 130, 185, 110, "Documents", ["text and fields", "with stable IDs"], "blue"),
        board.card(250, 130, 185, 110, "Analyse", ["tokenise", "normalise terms"], "teal"),
        board.card(470, 130, 185, 110, "Inverted index", ["term → postings", "fast candidate lookup"], "purple"),
        board.card(690, 130, 185, 110, "Score", ["rank candidates", "by query relevance"], "orange"),
        board.card(910, 130, 185, 110, "Results", ["best first", "evaluate with labels"], "green"),
    ]
    for left, right in zip(boxes, boxes[1:]):
        board.arrow(left.right(), right.left())
    board.card(195, 290, 730, 65, "Worked retrieval: 40 relevant among 50 shown; 60 relevant exist", ["precision 0.800 · recall 0.667 · F1 0.727"], "yellow", size=14)
    return board


def boolean_postings():
    board = Board(1100, 420, "Boolean retrieval uses sorted postings", "The lecture's two lists intersect to [1, 2, 4, 31]")
    left = board.card(55, 125, 430, 95, "List A", ["1 · 2 · 4 · 11 · 31"], "blue", size=20)
    right = board.card(615, 125, 430, 95, "List B", ["1 · 2 · 4 · 5 · 31"], "purple", size=20)
    middle = board.card(390, 265, 320, 90, "AND", ["1 · 2 · 4 · 31"], "green", size=20)
    board.arrow(left.bottom(), middle.left(0.35))
    board.arrow(right.bottom(), middle.right(0.35))
    board.text(550, 392, "Advance the pointer on the smaller document ID", 14, "grey")
    return board


def tolerant_terms():
    board = Board(1120, 430, "A typo changes the lookup path", "Normalise first, then generate and check a small candidate set")
    query = board.card(35, 140, 190, 85, "Query", ["cart"], "blue", size=20)
    normal = board.card(270, 140, 190, 85, "Normalise", ["case and tokens"], "teal")
    candidate = board.card(505, 140, 235, 85, "Candidates", ["k-grams or prefix", "narrow the terms"], "purple")
    distance = board.card(785, 140, 300, 85, "Edit distance", ["cat → cart = 1", "cat → dog = 3"], "green")
    for a, b in [(query, normal), (normal, candidate), (candidate, distance)]:
        board.arrow(a.right(), b.left())
    board.card(180, 285, 760, 90, "Different jobs", ["Exact lookup uses the dictionary; tolerant search expands terms.", "Keep expansion bounded so a typo cannot scan the whole vocabulary."], "yellow", size=14)
    return board


def index_compression():
    board = Board(1160, 460, "Build in blocks, store smaller gaps", "The lecture's 10 GB construction and three-posting compression examples")
    source = board.card(30, 120, 230, 95, "10 GB term-ID pairs", ["1 GB memory budget"], "blue")
    blocks = board.card(310, 120, 230, 95, "10 sorted blocks", ["merge into an index"], "teal")
    index = board.card(590, 120, 230, 95, "Postings", ["5 · 130 · 132"], "purple")
    gaps = board.card(870, 120, 260, 95, "Gaps", ["5 · 125 · 2"], "orange")
    for a, b in [(source, blocks), (blocks, index), (index, gaps)]:
        board.arrow(a.right(), b.left())
    board.card(240, 290, 680, 105, "Variable-byte result", ["0x85 · 0xFD · 0x82 = 3 bytes", "Three 32-bit document IDs would take 12 bytes"], "green", size=16)
    board.text(580, 435, "This byte count covers only these IDs; a full index has metadata and other fields", 12, "grey")
    return board


def vector_space():
    board = Board(1150, 450, "Weighted terms become a ranking", "Session 5's two worked numbers: tf-idf 3.0 and cosine 0.816")
    term = board.card(30, 120, 250, 95, "Corpus statistic", ["N = 1000", "df = 100"], "blue", size=16)
    idf = board.card(330, 120, 220, 95, "IDF", ["log10(1000 / 100)", "= 1.0"], "teal", size=16)
    weight = board.card(600, 120, 220, 95, "Term weight", ["tf = 3", "tf × idf = 3.0"], "purple", size=16)
    rank = board.card(870, 120, 250, 95, "Rank by cosine", ["q = [1,0,1,0]", "d = [1,1,1,0]"], "orange", size=14)
    for a, b in [(term, idf), (idf, weight), (weight, rank)]:
        board.arrow(a.right(), b.left())
    board.card(265, 290, 620, 95, "Cosine = 2 / (√2 × √3) = 0.816", ["Vector length is normalised before comparing documents"], "green", size=17)
    return board


def document_grouping():
    board = Board(1120, 465, "Two ways to organise documents", "Supervised classification uses labels; clustering discovers groups")
    board.group(30, 115, 505, 290, "Known categories", "blue")
    board.group(585, 115, 505, 290, "Unknown groups", "purple")
    sport = board.card(65, 180, 200, 75, "Sport", ["centroid cosine 0.7"], "blue", size=16)
    politics = board.card(300, 180, 200, 75, "Politics", ["centroid cosine 0.4"], "orange", size=16)
    result = board.card(160, 300, 250, 65, "Assign sport", ["nearest labelled centroid"], "green")
    board.arrow(sport.bottom(), result.left(0.3))
    board.arrow(politics.bottom(), result.right(0.3))
    assign = board.card(625, 180, 190, 75, "Assign points", ["nearest centre"], "purple")
    recompute = board.card(860, 180, 190, 75, "Update centres", ["mean of members"], "teal")
    board.arrow(assign.right(), recompute.left())
    board.card(695, 300, 310, 65, "Repeat until stable", ["k-means has no class labels"], "yellow")
    return board


def ranking_evaluation():
    board = Board(1120, 450, "Ranking order changes the score", "The lecture's relevant results at ranks 1, 3 and 5")
    ranks = [("1", "relevant", "P@1 = 1.000", "green"), ("2", "irrelevant", "", "grey"),
             ("3", "relevant", "P@3 = 0.667", "green"), ("4", "irrelevant", "", "grey"),
             ("5", "relevant", "P@5 = 0.600", "green")]
    for i, (rank, label, precision, color) in enumerate(ranks):
        board.card(30 + i * 220, 135, 195, 115, f"Rank {rank}", [label, precision], color, size=14)
    board.card(220, 310, 680, 80, "Average Precision = (1 + 2/3 + 3/5) / 3 = 0.756", ["Order matters even when the same three documents are retrieved"], "blue", size=14)
    return board


def web_search():
    board = Board(1140, 470, "Web search adds an open, changing corpus", "Crawl, deduplicate and rank for different query intents")
    board.card(35, 125, 310, 115, "Web pages", ["billions of possible URLs", "links, copies and spam"], "blue")
    board.card(415, 125, 310, 115, "Search pipeline", ["crawl → index → serve", "content + link signals"], "purple")
    board.card(795, 125, 310, 115, "User intent", ["information · navigation", "transaction"], "green")
    board.arrow((345, 180), (415, 180))
    board.arrow((725, 180), (795, 180))
    board.card(230, 300, 680, 100, "Index overlap estimate", ["p_A = 0.4 · p_B = 0.5", "|A| / |B| = p_B / p_A = 1.25"], "yellow", size=18)
    return board


def web_crawling():
    board = Board(1150, 475, "Scale a crawler across hosts", "Keep each host polite while the frontier discovers new pages")
    seed = board.card(30, 125, 220, 90, "Seed URLs", ["start discovery"], "blue")
    frontier = board.card(300, 125, 230, 90, "URL frontier", ["host queues", "deduplicate URLs"], "purple")
    fetch = board.card(580, 125, 230, 90, "Fetch", ["robots rules", "per-host delay"], "orange")
    index = board.card(860, 125, 260, 90, "Index", ["extract links", "refresh content"], "green")
    for left, right in [(seed, frontier), (frontier, fetch), (fetch, index)]:
        board.arrow(left.right(), right.left())
    board.card(205, 300, 740, 100, "Worked capacity", ["500 active hosts × 1 request per host per second", "≈ 500 pages per second before bottlenecks"], "teal", size=17)
    return board


def link_analysis():
    board = Board(1120, 465, "A link is a signal, not a verdict", "One PageRank update on a three-page graph")
    a = board.card(50, 130, 205, 95, "Page A", ["rank 1/3", "links to P"], "blue")
    b = board.card(50, 275, 205, 95, "Page B", ["rank 1/3", "links to P"], "purple")
    p = board.card(345, 200, 220, 100, "Page P", ["receives A and B", "links to A"], "green")
    board.arrow(a.right(), p.left(0.25))
    board.arrow(b.right(), p.left(0.75))
    board.card(650, 145, 430, 165, "Random-surfer update", ["d = 0.85; N = 3", "base = (1 − d) / N = 0.05", "P next = 0.05 + 0.85 × 2/3", "= 0.617"], "yellow", size=17)
    board.text(565, 407, "Iterate to convergence; combine links with content and abuse checks", 14, "grey")
    return board


def hub_authority():
    board = Board(1120, 445, "HITS reinforces hubs and authorities", "A hub points to useful authorities; an authority is pointed to by useful hubs")
    h1 = board.card(30, 130, 215, 95, "Hub H1", ["links to A1, A2", "initial hub score 1"], "blue")
    h2 = board.card(30, 275, 215, 95, "Hub H2", ["links to A1", "initial hub score 1"], "purple")
    a1 = board.card(395, 130, 215, 95, "Authority A1", ["incoming from H1, H2", "new authority 2"], "green")
    a2 = board.card(395, 275, 215, 95, "Authority A2", ["incoming from H1", "new authority 1"], "teal")
    board.arrow(h1.right(), a1.left())
    board.arrow(h1.right(0.75), a2.left(0.25))
    board.arrow(h2.right(0.25), a1.left(0.75))
    board.card(680, 175, 405, 140, "Updated hub scores", ["H1 = A1 + A2 = 3", "H2 = A1 = 2", "normalise after each full round"], "yellow", size=17)
    return board


def cross_language():
    board = Board(1160, 480, "Bridge a query and documents in different languages", "Three routes trade translation work against ambiguity and model dependence")
    board.card(35, 125, 345, 160, "Translate the query", ["one short text", "cheap to process", "banco may mean bank or bench"], "blue", size=17)
    board.card(405, 125, 345, 160, "Translate documents", ["1,000 texts in the example", "reusable at query time", "preserve original evidence"], "purple", size=17)
    board.card(775, 125, 345, 160, "Shared embeddings", ["encode 1,000 docs + query", "compare in one vector space", "validate each language pair"], "teal", size=17)
    board.card(235, 340, 690, 85, "Language-aware processing still matters", ["tokenisation · morphology · scripts · names · relevance judgements"], "yellow", size=17)
    return board


def clip_training():
    board = Board(1150, 465, "Learn one space for image and text", "Contrastive training separates matching and mismatched pairs")
    image = board.card(30, 135, 255, 105, "Image encoder", ["photo → image vector", "train on paired data"], "blue")
    caption = board.card(30, 290, 255, 105, "Text encoder", ["caption → text vector", "train jointly"], "purple")
    match = board.card(390, 140, 315, 100, "Matching pair", ["raise cosine similarity", "positive training example"], "green")
    mismatch = board.card(390, 290, 315, 100, "Mismatched pair", ["lower relative similarity", "negative training example"], "orange")
    board.arrow(image.right(), match.left())
    board.arrow(caption.right(), match.left(0.75))
    board.arrow(image.right(0.75), mismatch.left(0.2))
    board.card(790, 190, 330, 130, "At retrieval time", ["encode a text query", "rank indexed image vectors", "validate results with labels"], "yellow", size=17)
    return board


def image_cosine():
    board = Board(1120, 425, "The lecture's cross-modal cosine", "These four-dimensional vectors are illustrative, not model outputs")
    q = board.card(40, 130, 325, 95, "Text vector q", ["[1, 0, 1, 0]", "length √2"], "blue", size=18)
    d = board.card(755, 130, 325, 95, "Image vector d", ["[1, 1, 1, 0]", "length √3"], "purple", size=18)
    score = board.card(350, 275, 420, 95, "Cosine similarity", ["dot product = 2", "2 / (√2 × √3) = 0.816"], "green", size=18)
    board.arrow(q.bottom(), score.left())
    board.arrow(d.bottom(), score.right())
    return board


def recommender_methods():
    board = Board(1140, 465, "Personalised retrieval uses more than one signal", "History can act as an implicit query, but new users and items need a fallback")
    board.card(30, 130, 335, 155, "Collaborative", ["similar users or items", "rating and interaction matrix", "cold start with no history"], "blue", size=17)
    board.card(402, 130, 335, 155, "Content-based", ["item features match profile", "works for a described new item", "can narrow discovery"], "purple", size=17)
    board.card(774, 130, 335, 155, "Hybrid", ["combine candidate sources", "rank for current context", "check diversity and utility"], "green", size=17)
    board.card(220, 340, 700, 75, "Weighted-neighbour example", ["(0.8 × 4 + 0.6 × 5) / (0.8 + 0.6) = 4.43"], "yellow", size=18)
    return board


def neural_retrieval():
    board = Board(1160, 480, "Retrieve broadly, rerank a smaller set", "Separate encoders give fast candidates; joint query-document scoring spends more work")
    lexical = board.card(30, 120, 250, 90, "BM25", ["exact term evidence", "fast lexical candidates"], "blue")
    dense = board.card(30, 245, 250, 90, "Dual encoder", ["separate vectors", "semantic candidates"], "purple")
    fusion = board.card(410, 185, 270, 105, "Fusion", ["RRF merges ranks", "wider shortlist"], "teal")
    rerank = board.card(805, 185, 300, 105, "Cross encoder", ["joint query + document", "rerank the shortlist"], "orange")
    board.arrow(lexical.right(), fusion.left(0.25))
    board.arrow(dense.right(), fusion.left(0.75))
    board.arrow(fusion.right(), rerank.left())
    board.card(205, 355, 750, 80, "The six-document teaching example", ["BM25, toy dense and RRF: P@2 = 0.50", "intent-coverage reranker: P@2 = 1.00"], "green", size=17)
    return board


def question_bank_map():
    board = Board(1150, 530, "Thirty-six questions across the IR course", "The comprehensive bank repeats Q22–Q36, so each question appears once")
    rows = [
        ("Q1–4", "Boolean and preprocessing", "4 questions", "blue"),
        ("Q5–16", "Index and vector space", "12 questions", "teal"),
        ("Q17–21", "Grouping and evaluation", "5 questions", "purple"),
        ("Q22–26", "Web and links", "5 questions", "orange"),
        ("Q27–31", "Languages and modalities", "5 questions", "green"),
        ("Q32–36", "Neural IR and synthesis", "5 questions", "yellow"),
    ]
    for index, (number, topic, count, color) in enumerate(rows):
        x = 30 + (index % 3) * 375
        y = 130 + (index // 3) * 180
        board.card(x, y, 340, 125, number, [topic, count], color, size=17)
    board.text(575, 488, "Attempt first · reveal the worked answer · check the assumption", 15, "grey")
    return board


def midsem_marks():
    board = Board(1150, 540, "The six-question IR mid-semester paper", "Thirty marks across Boolean retrieval, text matching and indexing")
    rows = [
        ("Q1", "Skip pointers", "4 marks", "blue"),
        ("Q2", "Edit distance", "5 marks", "teal"),
        ("Q3", "Boolean query", "3 marks", "purple"),
        ("Q4", "Wildcard dictionary", "5 marks", "orange"),
        ("Q5", "Cosine ranking", "6 marks", "green"),
        ("Q6", "Compression", "7 marks", "yellow"),
    ]
    for index, (number, topic, marks, color) in enumerate(rows):
        x = 30 + (index % 3) * 375
        y = 125 + (index // 3) * 180
        board.card(x, y, 340, 125, number, [topic, marks], color, size=18)
    board.card(375, 485, 400, 45, "Total: 30 marks", [], "green", size=17)
    return board


BOARDS = {"ir-pipeline": ir_pipeline, "boolean-postings": boolean_postings, "tolerant-terms": tolerant_terms, "index-compression": index_compression, "vector-space": vector_space, "document-grouping": document_grouping, "ranking-evaluation": ranking_evaluation, "web-search": web_search, "web-crawling": web_crawling, "link-analysis": link_analysis, "hub-authority": hub_authority, "cross-language": cross_language, "clip-training": clip_training, "image-cosine": image_cosine, "recommender-methods": recommender_methods, "neural-retrieval": neural_retrieval, "question-bank-map": question_bank_map, "midsem-marks": midsem_marks}


if __name__ == "__main__":
    for name, build in BOARDS.items():
        print(build().save(OUT / f"{name}.svg"))
