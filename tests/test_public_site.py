"""Small public-demo artifact checks; no server or pretrained weights required."""
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.footer = False
        self.paper_links = []
        self.all_pdf_links = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "footer":
            self.footer = True
        if tag == "a" and attributes.get("href", "").endswith(".pdf"):
            self.all_pdf_links.append(attributes)
            if self.footer:
                self.paper_links.append(attributes)

    def handle_endtag(self, tag):
        if tag == "footer":
            self.footer = False


def test_paper_is_a_footer_link_not_a_panel():
    page = (ROOT / "web/index.html").read_text()
    parser = Links()
    parser.feed(page)
    assert len(parser.all_pdf_links) == 1
    assert len(parser.paper_links) == 1
    assert parser.paper_links[0]["href"] == "./paper.pdf"
    assert "paper-link" in parser.paper_links[0]["class"].split()
    assert (ROOT / "paper/paper.pdf").read_bytes().startswith(b"%PDF-")


def test_single_checkpoint_claim_is_present():
    page = (ROOT / "web/index.html").read_text()
    assert "149.6M parameters" in page
    assert "No generative LLM" in page
    assert "116,091 passages" in page
