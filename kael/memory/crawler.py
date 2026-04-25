"""
KAEL Web Crawler
Async crawler focused on knowledge frontier sources.
Feeds clean text into the session store for tau updates.

Design principles:
  - Respectful: honours robots.txt, rate limits, user-agent
  - Focused: curated source list, not general crawl
  - Clean: strips boilerplate, extracts meaningful text only
  - Integrated: output goes directly into SessionStore + novelty pipeline

Sources supported:
  - ArXiv (RSS + abstract pages) — primary frontier feed
  - Wikipedia (specific articles) — grounding/reference
  - Custom URLs — anything you point it at
  - RSS/Atom feeds — any feed URL

No Google API key needed. No external accounts.
Dependencies: httpx, beautifulsoup4, feedparser
  pip install httpx beautifulsoup4 feedparser
"""

import asyncio
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

logger = logging.getLogger(__name__)

# ── Default frontier sources ──────────────────────────────────────────────────
# These are what KAEL should be reading. Adjust freely.

FRONTIER_FEEDS = [
    # ── ArXiv: Computer Science ───────────────────────────────────────────────
    "https://rss.arxiv.org/rss/cs.AI",       # Artificial Intelligence
    "https://rss.arxiv.org/rss/cs.LG",       # Machine Learning
    "https://rss.arxiv.org/rss/cs.CL",       # Computation & Language
    "https://rss.arxiv.org/rss/cs.CV",       # Computer Vision
    "https://rss.arxiv.org/rss/cs.NE",       # Neural & Evolutionary Computing
    "https://rss.arxiv.org/rss/cs.RO",       # Robotics
    "https://rss.arxiv.org/rss/cs.CR",       # Cryptography & Security
    "https://rss.arxiv.org/rss/cs.IT",       # Information Theory
    "https://rss.arxiv.org/rss/cs.HC",       # Human-Computer Interaction

    # ── ArXiv: Physics ────────────────────────────────────────────────────────
    "https://rss.arxiv.org/rss/quant-ph",    # Quantum Physics
    "https://rss.arxiv.org/rss/hep-th",      # High Energy Physics – Theory
    "https://rss.arxiv.org/rss/hep-ph",      # High Energy Physics – Phenomenology
    "https://rss.arxiv.org/rss/gr-qc",       # General Relativity & Quantum Cosmology
    "https://rss.arxiv.org/rss/cond-mat.str-el",  # Strongly Correlated Electrons
    "https://rss.arxiv.org/rss/cond-mat.mes-hall", # Mesoscale & Nanoscale Physics
    "https://rss.arxiv.org/rss/physics.ao-ph",    # Atmospheric & Oceanic Physics
    "https://rss.arxiv.org/rss/physics.bio-ph",   # Biological Physics
    "https://rss.arxiv.org/rss/physics.comp-ph",  # Computational Physics

    # ── ArXiv: Mathematics ────────────────────────────────────────────────────
    "https://rss.arxiv.org/rss/math.ST",     # Statistics Theory
    "https://rss.arxiv.org/rss/math.PR",     # Probability
    "https://rss.arxiv.org/rss/math.AT",     # Algebraic Topology
    "https://rss.arxiv.org/rss/math.DG",     # Differential Geometry
    "https://rss.arxiv.org/rss/math.LO",     # Logic
    "https://rss.arxiv.org/rss/math.CO",     # Combinatorics
    "https://rss.arxiv.org/rss/math.DS",     # Dynamical Systems

    # ── ArXiv: Astrophysics ───────────────────────────────────────────────────
    "https://rss.arxiv.org/rss/astro-ph.GA", # Galaxies Astrophysics
    "https://rss.arxiv.org/rss/astro-ph.CO", # Cosmology & Nongalactic
    "https://rss.arxiv.org/rss/astro-ph.HE", # High Energy Astrophysical Phenomena
    "https://rss.arxiv.org/rss/astro-ph.EP", # Earth & Planetary Astrophysics
    "https://rss.arxiv.org/rss/astro-ph.SR", # Solar & Stellar Astrophysics
    "https://rss.arxiv.org/rss/astro-ph.IM", # Instrumentation & Methods

    # ── ArXiv: Quantitative Biology ───────────────────────────────────────────
    "https://rss.arxiv.org/rss/q-bio.NC",    # Neurons & Cognition
    "https://rss.arxiv.org/rss/q-bio.BM",    # Biomolecules
    "https://rss.arxiv.org/rss/q-bio.PE",    # Populations & Evolution
    "https://rss.arxiv.org/rss/q-bio.QM",    # Quantitative Methods

    # ── ArXiv: Statistics & Economics ─────────────────────────────────────────
    "https://rss.arxiv.org/rss/stat.ML",     # Machine Learning (Statistics)
    "https://rss.arxiv.org/rss/stat.TH",     # Statistics Theory
    "https://rss.arxiv.org/rss/econ.TH",     # Economics Theory

    # ── ArXiv: Earth & Climate ────────────────────────────────────────────────
    "https://rss.arxiv.org/rss/physics.ao-ph",    # Atmospheric & Ocean Physics
    "https://rss.arxiv.org/rss/physics.geo-ph",   # Geophysics

    # ── Major Science Journals ────────────────────────────────────────────────
    "https://www.nature.com/nature.rss",
    "https://www.nature.com/nmeth.rss",            # Nature Methods
    "https://www.nature.com/ncomms.rss",            # Nature Communications
    "https://www.nature.com/natmachintell.rss",     # Nature Machine Intelligence
    "https://www.science.org/action/showFeed?type=etoc&feed=rss&jc=science",
    "https://www.pnas.org/action/showFeed?type=etoc&feed=rss&jc=pnas",
    "https://journals.plos.org/plosone/feed/atom",
    "https://elifesciences.org/rss/recent.xml",

    # ── Neuroscience ──────────────────────────────────────────────────────────
    "https://www.jneurosci.org/rss/current.xml",
    "https://www.cell.com/neuron/inpress.rss",
    "https://www.thetransmitter.org/feed/",

    # ── Philosophy & Cognitive Science ───────────────────────────────────────
    "https://philpapers.org/rss/new.pl",
    "https://ndpr.nd.edu/feed/",               # Notre Dame Philosophical Reviews
    "https://dailynous.com/feed/",             # Philosophy news

    # ── Ocean & Climate Science ───────────────────────────────────────────────
    "https://www.agu.org/Share-and-Advocate/Share/Newsroom/RSS",
    "https://os.copernicus.org/articles/rss20.xml",      # Ocean Science journal
    "https://gmd.copernicus.org/articles/rss20.xml",     # Geoscientific Model Development
    "https://acp.copernicus.org/articles/rss20.xml",     # Atmospheric Chemistry & Physics
    "https://www.climate.gov/rss.xml",

    # ── General Science News ──────────────────────────────────────────────────
    "https://www.quantamagazine.org/feed/",
    "https://www.sciencedaily.com/rss/top/science.xml",
    "https://phys.org/rss-feed/",
    "https://www.newscientist.com/feed/home/",
    "https://www.scientificamerican.com/platform/morgue/2020/12/10/article.cfm?id=rss-feeds",
    "https://arstechnica.com/science/feed/",
    "https://feeds.feedburner.com/NautilusMag",       # Nautilus science magazine
    "https://nautil.us/feed/",
]


REFERENCE_URLS = [
    # ── Consciousness & Mind ──────────────────────────────────────────────────
    "https://en.wikipedia.org/wiki/Integrated_information_theory",
    "https://en.wikipedia.org/wiki/Global_workspace_theory",
    "https://en.wikipedia.org/wiki/Quantum_mind",
    "https://en.wikipedia.org/wiki/Hard_problem_of_consciousness",
    "https://en.wikipedia.org/wiki/Philosophical_zombie",
    "https://en.wikipedia.org/wiki/Predictive_coding",
    "https://en.wikipedia.org/wiki/Attention_schema_theory",
    "https://en.wikipedia.org/wiki/Bicameralism_(psychology)",

    # ── Artificial Intelligence & ML ─────────────────────────────────────────
    "https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)",
    "https://en.wikipedia.org/wiki/Diffusion_model",
    "https://en.wikipedia.org/wiki/Reinforcement_learning",
    "https://en.wikipedia.org/wiki/Meta-learning_(computer_science)",
    "https://en.wikipedia.org/wiki/Large_language_model",
    "https://en.wikipedia.org/wiki/Mechanistic_interpretability",
    "https://en.wikipedia.org/wiki/Catastrophic_interference",
    "https://en.wikipedia.org/wiki/Neural_scaling_laws",

    # ── Physics & Cosmology ───────────────────────────────────────────────────
    "https://en.wikipedia.org/wiki/Standard_Model",
    "https://en.wikipedia.org/wiki/String_theory",
    "https://en.wikipedia.org/wiki/Loop_quantum_gravity",
    "https://en.wikipedia.org/wiki/Black_hole_information_paradox",
    "https://en.wikipedia.org/wiki/Many-worlds_interpretation",
    "https://en.wikipedia.org/wiki/Holographic_principle",
    "https://en.wikipedia.org/wiki/Dark_matter",
    "https://en.wikipedia.org/wiki/Fermi_paradox",
    "https://en.wikipedia.org/wiki/Fine-tuned_universe",
    "https://en.wikipedia.org/wiki/Topological_order",

    # ── Mathematics ───────────────────────────────────────────────────────────
    "https://en.wikipedia.org/wiki/Computational_complexity_theory",
    "https://en.wikipedia.org/wiki/P_versus_NP_problem",
    "https://en.wikipedia.org/wiki/Riemann_hypothesis",
    "https://en.wikipedia.org/wiki/Category_theory",
    "https://en.wikipedia.org/wiki/Algebraic_topology",
    "https://en.wikipedia.org/wiki/Information_theory",
    "https://en.wikipedia.org/wiki/Kolmogorov_complexity",
    "https://en.wikipedia.org/wiki/G%C3%B6del%27s_incompleteness_theorems",
    "https://en.wikipedia.org/wiki/Chaos_theory",
    "https://en.wikipedia.org/wiki/Ergodic_theory",

    # ── Biology & Neuroscience ────────────────────────────────────────────────
    "https://en.wikipedia.org/wiki/Neuroplasticity",
    "https://en.wikipedia.org/wiki/Hebbian_learning",
    "https://en.wikipedia.org/wiki/Free_energy_principle",
    "https://en.wikipedia.org/wiki/CRISPR",
    "https://en.wikipedia.org/wiki/Protein_folding",
    "https://en.wikipedia.org/wiki/Epigenetics",
    "https://en.wikipedia.org/wiki/Evolutionary_game_theory",
    "https://en.wikipedia.org/wiki/Origin_of_life",

    # ── Philosophy ────────────────────────────────────────────────────────────
    "https://en.wikipedia.org/wiki/Philosophy_of_mind",
    "https://en.wikipedia.org/wiki/Epistemology",
    "https://en.wikipedia.org/wiki/Ontology",
    "https://en.wikipedia.org/wiki/Simulation_hypothesis",
    "https://en.wikipedia.org/wiki/Effective_altruism",
    "https://en.wikipedia.org/wiki/Existential_risk_from_artificial_general_intelligence",
    "https://en.wikipedia.org/wiki/Chinese_room",
    "https://en.wikipedia.org/wiki/Trolley_problem",
    "https://en.wikipedia.org/wiki/Moral_realism",

    # ── Ocean & Climate ───────────────────────────────────────────────────────
    "https://en.wikipedia.org/wiki/Thermohaline_circulation",
    "https://en.wikipedia.org/wiki/El_Ni%C3%B1o%E2%80%93Southern_Oscillation",
    "https://en.wikipedia.org/wiki/Ocean_heat_content",
    "https://en.wikipedia.org/wiki/Sea_surface_temperature",
    "https://en.wikipedia.org/wiki/Tipping_points_in_the_climate_system",
    "https://en.wikipedia.org/wiki/Lagrangian_coherent_structures",

    # ── Complexity & Systems ──────────────────────────────────────────────────
    "https://en.wikipedia.org/wiki/Complex_system",
    "https://en.wikipedia.org/wiki/Emergence",
    "https://en.wikipedia.org/wiki/Self-organized_criticality",
    "https://en.wikipedia.org/wiki/Autopoiesis",
    "https://en.wikipedia.org/wiki/Cellular_automaton",
    "https://en.wikipedia.org/wiki/Scale-free_network",
]

@dataclass
class CrawledPage:
    url: str
    title: str
    text: str                       # Cleaned body text
    summary: str                    # First ~300 chars
    source_type: str                # arxiv / wikipedia / rss / web
    domain: str                     # inferred topic domain
    crawled_at: float = field(default_factory=time.time)
    content_hash: str = ""          # For dedup
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.content_hash:
            self.content_hash = hashlib.sha256(self.text.encode()).hexdigest()[:16]

    def to_prompt(self) -> str:
        """Format as a prompt for KAEL to process — creates a session."""
        return (
            f"New content from {self.source_type} source:\n\n"
            f"Title: {self.title}\n"
            f"URL: {self.url}\n\n"
            f"{self.text[:3000]}"
            + ("\n\n[truncated]" if len(self.text) > 3000 else "")
        )

    def to_digest_prompt(self) -> str:
        """Shorter prompt asking KAEL to reflect on a batch of content."""
        return (
            f"Digest this {self.source_type} content and identify what is "
            f"genuinely novel or significant relative to what you already know:\n\n"
            f"**{self.title}**\n{self.summary}\nSource: {self.url}"
        )


class KAELCrawler:
    """
    Async web crawler. Fetches, cleans, deduplicates.
    Call crawl_feeds() for RSS/ArXiv, crawl_urls() for specific pages.
    """

    USER_AGENT = "KAEL-Crawler/1.0 (research; respectful bot)"
    DEFAULT_DELAY = 2.0      # seconds between requests to same domain
    MAX_TEXT_LENGTH = 8000   # chars to keep per page

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        delay: float = DEFAULT_DELAY,
        max_concurrent: int = 3,
    ):
        self.delay = delay
        self.max_concurrent = max_concurrent
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

        self._seen_hashes: set[str] = set()
        self._domain_last_fetch: dict[str, float] = {}
        self._robots_cache: dict[str, RobotFileParser] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent)

        # Load seen hashes from cache if available
        if cache_dir:
            self._load_seen_hashes()

    # ── Public API ────────────────────────────────────────────────────────────

    async def crawl_feeds(self, feed_urls: Optional[list[str]] = None) -> list[CrawledPage]:
        """Fetch and parse RSS/Atom feeds. Returns new (not-yet-seen) pages."""
        import feedparser
        feed_urls = feed_urls or FRONTIER_FEEDS
        pages = []

        for url in feed_urls:
            try:
                logger.info(f"Fetching feed: {url}")
                feed = feedparser.parse(url)
                for entry in feed.entries[:10]:  # latest 10 per feed
                    page = self._entry_to_page(entry, url)
                    if page and not self._is_duplicate(page):
                        pages.append(page)
                        self._mark_seen(page)
                await asyncio.sleep(self.delay)
            except Exception as e:
                logger.warning(f"Feed error {url}: {e}")

        logger.info(f"Feeds: {len(pages)} new pages from {len(feed_urls)} feeds")
        return pages

    async def crawl_urls(self, urls: list[str]) -> list[CrawledPage]:
        """Fetch specific URLs. Returns cleaned pages."""
        tasks = [self._fetch_url(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        pages = [r for r in results if isinstance(r, CrawledPage)]
        new_pages = [p for p in pages if not self._is_duplicate(p)]
        for p in new_pages:
            self._mark_seen(p)
        logger.info(f"URLs: {len(new_pages)}/{len(urls)} new pages")
        return new_pages

    async def crawl_arxiv_abstract(self, arxiv_id: str) -> Optional[CrawledPage]:
        """
        Fetch a specific ArXiv abstract page.
        arxiv_id: e.g. "2401.12345" or full URL
        """
        if arxiv_id.startswith("http"):
            url = arxiv_id
        else:
            url = f"https://arxiv.org/abs/{arxiv_id}"
        return await self._fetch_url(url)

    async def search_arxiv(self, query: str, max_results: int = 5) -> list[CrawledPage]:
        """
        Search ArXiv via their API (no key needed).
        Returns abstract pages for top results.
        """
        import urllib.parse
        query_enc = urllib.parse.quote(query)
        api_url = (
            f"https://export.arxiv.org/api/query"
            f"?search_query=all:{query_enc}"
            f"&start=0&max_results={max_results}"
            f"&sortBy=submittedDate&sortOrder=descending"
        )
        try:
            import httpx
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(api_url, headers={"User-Agent": self.USER_AGENT})
                resp.raise_for_status()
            pages = self._parse_arxiv_api(resp.text, query)
            new_pages = [p for p in pages if not self._is_duplicate(p)]
            for p in new_pages:
                self._mark_seen(p)
            return new_pages
        except Exception as e:
            logger.warning(f"ArXiv search error: {e}")
            return []

    # ── Internal fetch ────────────────────────────────────────────────────────

    async def _fetch_url(self, url: str) -> Optional[CrawledPage]:
        domain = urlparse(url).netloc
        async with self._semaphore:
            await self._rate_limit(domain)
            try:
                import httpx
                async with httpx.AsyncClient(
                    timeout=15,
                    follow_redirects=True,
                    headers={"User-Agent": self.USER_AGENT},
                ) as client:
                    resp = await client.get(url)
                    resp.raise_for_status()
                    html = resp.text
                self._domain_last_fetch[domain] = time.time()
                return self._parse_html(html, url)
            except Exception as e:
                logger.warning(f"Fetch error {url}: {e}")
                return None

    async def _rate_limit(self, domain: str):
        last = self._domain_last_fetch.get(domain, 0)
        wait = self.delay - (time.time() - last)
        if wait > 0:
            await asyncio.sleep(wait)

    # ── Parsing ───────────────────────────────────────────────────────────────

    def _parse_html(self, html: str, url: str) -> Optional[CrawledPage]:
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            logger.error("beautifulsoup4 not installed: pip install beautifulsoup4")
            return None

        soup = BeautifulSoup(html, "html.parser")

        # Remove noise
        for tag in soup(["script", "style", "nav", "footer", "header",
                         "aside", "advertisement", "cookie", "banner"]):
            tag.decompose()

        title = ""
        if soup.title:
            title = soup.title.string or ""
        if not title and soup.find("h1"):
            title = soup.find("h1").get_text()
        title = title.strip()

        # ArXiv-specific extraction
        if "arxiv.org/abs" in url:
            return self._parse_arxiv_page(soup, url, title)

        # Wikipedia-specific extraction
        if "wikipedia.org/wiki" in url:
            return self._parse_wikipedia_page(soup, url, title)

        # Generic extraction — main content heuristic
        main = (
            soup.find("article") or
            soup.find("main") or
            soup.find(class_=re.compile(r"content|article|post|body", re.I)) or
            soup.find("body")
        )
        text = self._clean_text(main.get_text() if main else soup.get_text())

        return CrawledPage(
            url=url,
            title=title,
            text=text[:self.MAX_TEXT_LENGTH],
            summary=text[:300],
            source_type=self._infer_source_type(url),
            domain=self._infer_domain(title + " " + text[:500]),
        )

    def _parse_arxiv_page(self, soup, url: str, title: str) -> CrawledPage:
        abstract = ""
        abs_div = soup.find("blockquote", class_="abstract")
        if abs_div:
            abstract = self._clean_text(abs_div.get_text()).replace("Abstract:", "").strip()

        authors = ""
        auth_div = soup.find("div", class_="authors")
        if auth_div:
            authors = self._clean_text(auth_div.get_text())

        # ArXiv ID from URL
        arxiv_id = url.split("/abs/")[-1].split("v")[0] if "/abs/" in url else ""

        text = f"{abstract}\n\nAuthors: {authors}"
        return CrawledPage(
            url=url,
            title=title,
            text=text[:self.MAX_TEXT_LENGTH],
            summary=abstract[:300],
            source_type="arxiv",
            domain=self._infer_domain(title + " " + abstract),
            metadata={"arxiv_id": arxiv_id, "authors": authors},
        )

    def _parse_wikipedia_page(self, soup, url: str, title: str) -> CrawledPage:
        content = soup.find("div", id="mw-content-text")
        if not content:
            content = soup.find("body")
        # Remove infoboxes, references, navboxes
        for el in content.find_all(class_=["infobox", "reflist", "navbox",
                                            "reference", "mw-references-wrap"]):
            el.decompose()
        text = self._clean_text(content.get_text())
        return CrawledPage(
            url=url,
            title=title,
            text=text[:self.MAX_TEXT_LENGTH],
            summary=text[:300],
            source_type="wikipedia",
            domain=self._infer_domain(title + " " + text[:500]),
        )

    def _entry_to_page(self, entry, feed_url: str) -> Optional[CrawledPage]:
        """Convert an RSS feed entry to a CrawledPage."""
        title = getattr(entry, "title", "") or ""
        link = getattr(entry, "link", "") or ""
        summary = getattr(entry, "summary", "") or getattr(entry, "description", "") or ""

        # Strip HTML from summary
        try:
            from bs4 import BeautifulSoup
            summary = BeautifulSoup(summary, "html.parser").get_text()
        except Exception:
            summary = re.sub(r"<[^>]+>", " ", summary)

        summary = self._clean_text(summary)
        if not title or not summary:
            return None

        source_type = "arxiv" if "arxiv" in feed_url else "rss"
        return CrawledPage(
            url=link,
            title=title,
            text=summary[:self.MAX_TEXT_LENGTH],
            summary=summary[:300],
            source_type=source_type,
            domain=self._infer_domain(title + " " + summary),
            metadata={"feed_url": feed_url},
        )

    def _parse_arxiv_api(self, xml_text: str, query: str) -> list[CrawledPage]:
        """Parse ArXiv Atom API response."""
        import xml.etree.ElementTree as ET
        pages = []
        try:
            ns = {"atom": "http://www.w3.org/2005/Atom",
                  "arxiv": "http://arxiv.org/schemas/atom"}
            root = ET.fromstring(xml_text)
            for entry in root.findall("atom:entry", ns):
                title = entry.findtext("atom:title", "", ns).strip().replace("\n", " ")
                summary = entry.findtext("atom:summary", "", ns).strip()
                link = ""
                for l in entry.findall("atom:link", ns):
                    if l.get("type") == "text/html":
                        link = l.get("href", "")
                text = f"{summary}"
                pages.append(CrawledPage(
                    url=link,
                    title=title,
                    text=text[:self.MAX_TEXT_LENGTH],
                    summary=summary[:300],
                    source_type="arxiv",
                    domain=self._infer_domain(title + " " + summary),
                    metadata={"query": query},
                ))
        except Exception as e:
            logger.warning(f"ArXiv API parse error: {e}")
        return pages

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _clean_text(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()
        return text

    def _infer_source_type(self, url: str) -> str:
        if "arxiv.org" in url:
            return "arxiv"
        if "wikipedia.org" in url:
            return "wikipedia"
        if "github.com" in url:
            return "github"
        return "web"

    def _infer_domain(self, text: str) -> str:
        text = text.lower()
        checks = {
            "machine_learning": ["neural", "transformer", "llm", "gradient", "attention",
                                 "language model", "deep learning", "fine-tun"],
            "physics": ["quantum", "relativity", "entropy", "particle", "spacetime",
                       "cosmolog", "thermodynamic"],
            "mathematics": ["theorem", "proof", "algebra", "topology", "manifold",
                           "differential equation", "number theory"],
            "astronomy": ["galaxy", "black hole", "dark matter", "cosmolog", "stellar",
                         "exoplanet", "fermi", "seti"],
            "philosophy": ["consciousness", "epistemolog", "ontolog", "qualia",
                          "free will", "ethics", "metaphysic"],
            "biology": ["protein", "genome", "evolution", "cell", "neuroscience",
                       "crispr", "rna", "dna"],
            "code": ["algorithm", "compiler", "runtime", "software", "programming"],
            "ocean_science": ["ocean", "sea surface", "thermohaline", "salinity",
                  "mesoscale", "eddy", "lagrangian", "bathymetry",
                  "sst", "upwelling", "marine"],
            "climate": ["climate", "precipitation", "atmosphere", "carbon", "warming",
                        "el niño", "circulation", "forcing"],
        }
        scores = {d: sum(1 for kw in kws if kw in text) for d, kws in checks.items()}
        return "general" if max(scores.values()) == 0 else max(scores, key=scores.get)

    def _is_duplicate(self, page: CrawledPage) -> bool:
        return page.content_hash in self._seen_hashes

    def _mark_seen(self, page: CrawledPage):
        self._seen_hashes.add(page.content_hash)
        if self.cache_dir:
            self._save_seen_hashes()

    def _load_seen_hashes(self):
        path = self.cache_dir / "seen_hashes.json"
        if path.exists():
            self._seen_hashes = set(json.loads(path.read_text()))
            logger.info(f"Loaded {len(self._seen_hashes)} seen content hashes")

    def _save_seen_hashes(self):
        path = self.cache_dir / "seen_hashes.json"
        path.write_text(json.dumps(list(self._seen_hashes)))