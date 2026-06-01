import csv
import re
from pathlib import Path
from urllib.parse import urlparse, unquote

BASE_DIR = Path("data/interim/gdelt_event_context_daily")

# Tune these based on what you see in your URLs
NEGATIVE_PATH_KEYWORDS = [
    # --- SPORTS ---
    "sport","sports","football","soccer","nba","nfl","mlb","nhl","mma","ufc",
    "boxing","wrestling","tennis","golf","cricket","rugby","f1","formula-1",
    "motorsport","nascar","cycling","olympics","athletics","baseball",
    "basketball","hockey","esports","gaming","afcon","chelsea","hat-trick",

    # --- ENTERTAINMENT ---
    "entertainment","celebrity","celebrities","hollywood","bollywood",
    "movies","movie","film","tv","television","cinema","anime",
    "streaming","netflix","hulu","prime-video","amazon-prime","disney",
    "disney-plus","hbomax","spotify","music","album","song","songs",
    "concert","tour","festival","theatre","theater","broadway","oscars",
    "emmys","grammys","kardashian","royal-family","celeb",
    "showbiz","tvshowbiz","arts","art","magazine",

    # --- LIFESTYLE / POP CULTURE ---
    "lifestyle","fashion","beauty","makeup","skincare","hair","diet",
    "fitness","yoga","workout","gym","weightloss","wellness",
    "relationships","dating","wedding","weddings","sex","parenting",
    "horoscope","astrology","zodiac","tarot","jewelry",

    # --- FOOD / DINING ---
    "recipe","recipes","cooking","cook","baking","kitchen","restaurant",
    "food","cuisine","dining","mayo","mayonnaise","chocolate","cake",
    "dessert","wine","beer","cocktail","coffee","tea",
    "pizza","burger","mcdonalds","candy","salmon","oyster",

    # --- TRAVEL / TOURISM ---
    "holiday","holidays","vacation","tourism","hotel","hotels",
    "cruise","beach","airport-guide","airbnb",

    # --- TECH / GADGET REVIEWS ---
    "gadget","gadgets","smartphone","iphone","android",
    "laptop","tablet","camera","headphones","earbuds","tv-review",
    "gaming-console","ps5","xbox","nintendo","geforce",

    # --- CLICKBAIT / VIRAL ---
    "quiz","giveaway","contest","sweepstakes","lottery",
    "viral","meme","memes","funny","top-10","top10",
    "slideshow","gallery","photos","pictures","wallpaper",

    # --- CRIME / VIOLENCE ---
    "homicide","stabbing","stabbed","carjacking","kidnap","ransom",
    "rape","raped","raping","trafficking","paedophile","robbers","fraud",
    "school-shooting","missing-person",

    # --- DRUGS ---
    "cocaine","drug","cannabis","heroin","fentanyl","marijuana",
    "methamphetamine","meth","opioid",

    # --- PERSONAL LIFE / GOSSIP ---
    "obituary","obituaries","funeral","wedding-announcement","birth",
    "anniversary","birthday","biography","childhood",
    "mom","dad","son","daughter","wife","husband","boyfriend","girlfriend",
    "parents","father","baby","babies","pregnancy","graduates","woman",
    "porn","pornography","epstein","virgin",

    # --- RELIGION & SOCIAL ISSUES ---
    "church","mosque","synagogue","baptist","bishop","bishops","priests",
    "racism","antisemitism","gay","queer","gender","abortion","suicide",
    "holocaust","jewish","jews","spiritual",

    # --- GEOPOLITICS / CIVIL CONFLICT (non-supply-chain) ---
    "israel","gaza","palestine","israeli","palestinian","genocide",
    "somali","somalia","somaliland","migrants",

    # --- EDUCATION & WELFARE ---
    "university","teachers","childcare","child","children","schools",

    # --- HEALTH / MEDICAL ---
    "measles","dementia","diabetes","vaccination","covid",

    # --- ANIMALS ---
    "pet","pets","dog","dogs","cat","cats","elephant","horse","turtle","seals",

    # --- MISCELLANEOUS NOISE ---
    "anime","laundry","bicycle","swimming","hiking","museum","casino",
    "love","flower","balloon","monster","cult","demon","moon","legend",
    "motel","mortgage","bitcoin","cryptocurrency","crypto",
    "chatgpt","tiktok","motorcycle","season","sickle","honda",
    "podcast","scream","equaliser","roma",
]

# Skip standard category/tag/author navigation pages that don't contain article content
NEGATIVE_PATH_PATTERNS = [
    r"/tag/", r"/tags/", r"/category/", r"/author/", r"/gallery/", r"/video/", r"/podcast/"
]

BAD_EXTENSIONS = (
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".webp",
    ".mp4", ".mov", ".avi", ".zip", ".rar", ".7z"
)


def _url_search_text(p) -> str:
    """
    Build a normalized search string from parts of the URL so keywords match more reliably.
    Includes netloc + path + query. Decodes %xx and normalizes separators.
    """
    netloc = (p.netloc or "")
    path = unquote(p.path or "")
    query = unquote(p.query or "")

    full = f"{netloc} {path} {query}".lower()
    # normalize separators into spaces
    full = re.sub(r"[-_/]+", " ", full)
    # collapse whitespace
    full = re.sub(r"\s+", " ", full).strip()
    return full


def is_irrelevant_url(url: str) -> tuple[bool, str]:
    """
    Conservative URL-only filter.
    Returns (True, reason) if we should drop it.
    """
    url = (url or "").strip()
    if not url.startswith("http"):
        return True, "non_http"

    try:
        p = urlparse(url)

        # Extension check based on decoded path
        path_lower = unquote(p.path or "").lower()
        if path_lower.endswith(BAD_EXTENSIONS):
            return True, "bad_extension"

        # obvious section/category pages
        for pat in NEGATIVE_PATH_PATTERNS:
            if re.search(pat, (p.path or "").lower()):
                return True, f"neg_pattern:{pat}"

        # keyword match (netloc + path + query), with token + substring fallback
        full = _url_search_text(p)
        tokens = set(re.split(r"[^a-z0-9]+", full))

        for kw in NEGATIVE_PATH_KEYWORDS:
            kw_l = kw.lower().strip()
            if not kw_l:
                continue

            # Token-level match works best for single words (exact boundary check)
            if kw_l in tokens:
                return True, f"neg_kw_token:{kw_l}"

            # Substring match catches compound keywords like 'formula-1' that survive tokenization
            if kw_l in full:
                return True, f"neg_kw_substr:{kw_l}"

        return False, ""
    except Exception:
        # if parsing fails, keep it (conservative)
        return False, ""


def dedupe_and_filter_file(path: Path) -> None:
    deduped_path = path.with_name(path.stem + "_deduped.csv")
    filtered_path = path.with_name(path.stem + "_deduped_filtered.csv")

    seen = set()
    kept_deduped = 0
    dropped_dupes = 0

    kept_filtered = 0
    dropped_irrelevant = 0

    with open(path, "r", newline="", encoding="utf-8") as f_in, \
         open(deduped_path, "w", newline="", encoding="utf-8") as f_deduped, \
         open(filtered_path, "w", newline="", encoding="utf-8") as f_filtered:

        reader = csv.reader(f_in)
        w_deduped = csv.writer(f_deduped)
        w_filtered = csv.writer(f_filtered)

        header = next(reader, None)
        if header is None:
            return

        w_deduped.writerow(header)
        w_filtered.writerow(header)

        try:
            url_idx = header.index("sourceurl")
        except ValueError:
            print(f"WARNING: no 'sourceurl' column in {path}")
            return

        for row in reader:
            if not row or len(row) <= url_idx:
                continue

            url = (row[url_idx] or "").strip()
            if not url:
                continue

            # 1) dedupe by URL
            if url in seen:
                dropped_dupes += 1
                continue

            seen.add(url)
            w_deduped.writerow(row)
            kept_deduped += 1

            # 2) filter irrelevant by URL parts
            drop, _reason = is_irrelevant_url(url)
            if drop:
                dropped_irrelevant += 1
                continue

            w_filtered.writerow(row)
            kept_filtered += 1

    print(
        f"{path.name}: "
        f"deduped kept {kept_deduped:,}, dupes dropped {dropped_dupes:,} | "
        f"filtered kept {kept_filtered:,}, irrelevant dropped {dropped_irrelevant:,}"
    )
    print(f"  -> {deduped_path.name}")
    print(f"  -> {filtered_path.name}")



def main(target_date: str): 
    # 1. We use rglob to search recursively through all subfolders (Year/Month/Day)
    # We look specifically for the raw context file for that date
    search_pattern = f"**/{target_date}_event_context.csv"
    files = list(BASE_DIR.rglob(search_pattern))
    
    if not files:
        # Fallback: search for any context file and filter manually by name 
        # (useful if the file naming convention varies slightly)
        all_context_files = list(BASE_DIR.rglob("*_event_context.csv"))
        files = [f for f in all_context_files if target_date in f.name]

    if not files:
        print(f"No files found for date {target_date} in {BASE_DIR}")
        return

    print(f"Found {len(files)} file(s) for {target_date}. Starting filtering...")

    for path in sorted(files):
        # The dedupe_and_filter_file function already uses path.with_name()
        # which means it will save the new CSVs in the EXACT same folder as the input.
        dedupe_and_filter_file(path)

if __name__ == "__main__":
    # If run by itself, ask for input
    day = input("Select date (YYYYMMDD): ").strip()
    main(day)
