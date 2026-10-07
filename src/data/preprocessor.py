"""Text preprocessing module for SpamGuard."""
import re
import html
from typing import List, Dict, Any, Optional
from ..config import PreprocessingConfig, DEFAULT_CONFIG

class TextPreprocessor:
    """Production-grade text cleaner and normalizer for email and SMS text."""

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or DEFAULT_CONFIG.preprocessing
        
        # Compiled regex patterns for high-throughput preprocessing
        self.html_tag_pattern = re.compile(r"<[^>]+>")
        self.url_pattern = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
        self.email_pattern = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
        self.phone_pattern = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}")
        self.currency_pattern = re.compile(r"[$€£¥₹]")
        self.number_pattern = re.compile(r"\b\d+[\d,.]*\b")
        self.repeated_punct_pattern = re.compile(r"([!?,.]){2,}")
        self.whitespace_pattern = re.compile(r"\s+")

        # Standard English stop words
        self.stop_words = {
            "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
            "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
            "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
            "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
            "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have",
            "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers",
            "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've",
            "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me",
            "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on",
            "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over",
            "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't",
            "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them",
            "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll",
            "they're", "they've", "this", "those", "through", "to", "too", "under", "until",
            "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
            "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while",
            "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you",
            "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"
        }

    def extract_features(self, raw_text: str) -> Dict[str, Any]:
        """Extract statistical metadata indicators of spam from raw email text."""
        if not raw_text:
            return {
                "char_length": 0,
                "word_count": 0,
                "uppercase_ratio": 0.0,
                "exclamation_count": 0,
                "question_count": 0,
                "currency_count": 0,
                "url_count": 0,
            }
        
        char_length = len(raw_text)
        words = raw_text.split()
        word_count = len(words)
        uppercase_chars = sum(1 for c in raw_text if c.isupper())
        uppercase_ratio = uppercase_chars / char_length if char_length > 0 else 0.0
        exclamation_count = raw_text.count("!")
        question_count = raw_text.count("?")
        currency_count = len(self.currency_pattern.findall(raw_text))
        url_count = len(self.url_pattern.findall(raw_text))

        # Phishing Threat Detection Heuristics (Credential Harvesting, Urgency, Impersonation)
        phishing_indicators = []
        p_urgency = bool(re.search(r"(?i)\b(expire[s]?|expiring|expiration)\s+(in|within)\s+\d+\s*(hours?|hrs?|days?|mins?)\b", raw_text) or
                         re.search(r"(?i)\b(immediate(ly)?|urgent(ly)?|critical alert|action required|within 24 hours|within 48 hours)\b", raw_text))
        if p_urgency:
            phishing_indicators.append("urgent_deadline_pressure")

        p_consequence = bool(re.search(r"(?i)\b(permanent(ly)?\s+closure|closure of your account|data will be lost|account will be (closed|deleted|suspended|terminated|deactivated))\b", raw_text) or
                             re.search(r"(?i)\b(failure to\s+(update|verify|confirm|validate)|access\s+(will be|is)\s+(revoked|suspended|disabled))\b", raw_text))
        if p_consequence:
            phishing_indicators.append("account_closure_threat")

        p_credential_lure = bool(re.search(r"(?i)\b(keep\s+current\s+password|update\s+(your\s+)?password|reset\s+(your\s+)?password|verify\s+(your\s+)?(identity|account|credentials|email|login))\b", raw_text) or
                                 re.search(r"(?i)\b(click(ing)?\s+(on\s+)?(our\s+|the\s+)?(url|link|button)\s+below|log[\s-]?in\s+to\s+(keep|verify|retain|confirm))\b", raw_text))
        if p_credential_lure:
            phishing_indicators.append("credential_harvesting_lure")

        p_impersonation = bool(re.search(r"(?i)\b(it\s+service(\s+message)?|it\s+helpdesk|it\s+support\s+desk|system\s+administrator|office\s*365\s+team|security\s+team|admin\s+support)\b", raw_text))
        if p_impersonation:
            phishing_indicators.append("it_helpdesk_impersonation")

        p_financial_lure = bool(re.search(r"(?i)\b(eft\s+payment|payment\s+receipt|payment\s+confirmation|invoice\s*#?\s*\d+|remittance\s+advice|wire\s+transfer\s+receipt)\b", raw_text))
        p_attachment_lure = bool(re.search(r"(?i)\b(open\s+in\s+excel|download\s+(document|attachment|invoice|file)|view\s+(invoice|receipt)|\.(xlsx|xlsm|iso|exe|scr|vbs))\b", raw_text))
        if p_financial_lure and p_attachment_lure:
            phishing_indicators.append("fake_invoice_attachment_lure")

        # Scoring
        phishing_risk = 0.0
        if p_credential_lure and (p_urgency or p_consequence):
            phishing_risk = 0.96 if (p_urgency and p_consequence) else 0.88
            if p_impersonation:
                phishing_risk = min(0.99, phishing_risk + 0.03)
        elif p_financial_lure and p_attachment_lure:
            phishing_risk = 0.98
        elif p_consequence and p_urgency:
            phishing_risk = 0.85
        elif p_credential_lure:
            phishing_risk = 0.45

        return {
            "char_length": char_length,
            "word_count": word_count,
            "uppercase_ratio": round(uppercase_ratio, 4),
            "exclamation_count": exclamation_count,
            "question_count": question_count,
            "currency_count": currency_count,
            "url_count": url_count,
            "phishing_risk_score": round(phishing_risk, 4),
            "phishing_indicators": phishing_indicators,
        }

    def clean_text(self, text: str, remove_stopwords: bool = False) -> str:
        """Clean and normalize a single text string."""
        if not isinstance(text, str) or not text.strip():
            return ""

        # Decode HTML entities (&amp; -> &, &#39; -> ', etc.)
        cleaned = html.unescape(text)

        # Extract URLs inside href attributes before stripping tags
        if self.config.normalize_urls:
            cleaned = re.sub(r'<a\s+[^>]*?href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', r' \2 [URL] ', cleaned, flags=re.IGNORECASE)
            cleaned = self.url_pattern.sub(" [URL] ", cleaned)

        # Strip HTML tags
        if self.config.strip_html:
            cleaned = self.html_tag_pattern.sub(" ", cleaned)


        # Normalize Email addresses
        if self.config.normalize_emails:
            cleaned = self.email_pattern.sub(" [EMAIL] ", cleaned)

        # Normalize Phone numbers
        cleaned = self.phone_pattern.sub(" [PHONE] ", cleaned)

        # Normalize Currency symbols
        if self.config.normalize_currency:
            cleaned = self.currency_pattern.sub(" [CURRENCY] ", cleaned)

        # Normalize Numbers
        if self.config.normalize_numbers:
            cleaned = self.number_pattern.sub(" [NUMBER] ", cleaned)

        # Compress repeated punctuations (e.g., '!!!!' -> '!')
        cleaned = self.repeated_punct_pattern.sub(r"\1", cleaned)

        # Lowercase
        if self.config.lowercase:
            cleaned = cleaned.lower()

        # Remove excessive non-alphanumeric characters while preserving token placeholders
        if self.config.remove_special_chars:
            cleaned = re.sub(r"[^a-zA-Z0-9\s\[\]]", " ", cleaned)

        # Collapse whitespace
        cleaned = self.whitespace_pattern.sub(" ", cleaned).strip()

        # Optional stop-word removal
        if remove_stopwords:
            tokens = cleaned.split()
            tokens = [t for t in tokens if t not in self.stop_words or t.startswith("[")]
            cleaned = " ".join(tokens)

        return cleaned

    def clean_series(self, texts: List[str], remove_stopwords: bool = False) -> List[str]:
        """Clean a batch of text strings efficiently."""
        return [self.clean_text(t, remove_stopwords=remove_stopwords) for t in texts]
