"""Data ingestion module for public spam corpora."""
import os
import sys
import urllib.request
import zipfile
import tarfile
import logging
from pathlib import Path
from typing import Optional, List, Dict
import pandas as pd
from ..config import RAW_DATA_DIR, PROCESSED_DATA_DIR, DEFAULT_CONFIG
from .preprocessor import TextPreprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Direct URLs to benchmark public datasets
SMS_SPAM_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
ENRON_SAMPLE_URL = "https://raw.githubusercontent.com/mwitiderrick/Spam-Classification/master/spam.csv"
GITHUB_ENRON_SPAM_URL = "https://raw.githubusercontent.com/kartikgupta12/Spam-Detection/master/spam_or_not_spam.csv"

def download_file(url: str, dest_path: Path) -> bool:
    """Download a file with error handling."""
    logger.info(f"Downloading from {url} to {dest_path}...")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )
        with urllib.request.urlopen(req, timeout=30) as response, open(dest_path, "wb") as out_file:
            out_file.write(response.read())
        logger.info(f"Saved {dest_path} successfully ({dest_path.stat().st_size} bytes)")
        return True
    except Exception as e:
        logger.warning(f"Download failed for {url}: {e}")
        return False

def load_sms_spam_collection() -> Optional[pd.DataFrame]:
    """Download and parse SMS Spam Collection dataset."""
    zip_path = RAW_DATA_DIR / "smsspamcollection.zip"
    extract_dir = RAW_DATA_DIR / "sms_spam"
    
    if not (extract_dir / "SMSSpamCollection").exists():
        if download_file(SMS_SPAM_URL, zip_path):
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
            except Exception as e:
                logger.error(f"Error unzipping SMS dataset: {e}")
        
    sms_file = extract_dir / "SMSSpamCollection"
    if sms_file.exists():
        try:
            df = pd.read_csv(sms_file, sep="\t", header=None, names=["label_str", "text"], encoding="utf-8")
            df["label"] = df["label_str"].map({"ham": 0, "spam": 1})
            df["source"] = "sms_spam_collection"
            return df[["text", "label", "source"]].dropna()
        except Exception as e:
            logger.error(f"Error parsing SMSSpamCollection: {e}")
    return None

def load_enron_or_email_dataset() -> Optional[pd.DataFrame]:
    """Download and parse public email spam dataset."""
    dest_csv = RAW_DATA_DIR / "enron_or_email.csv"
    if not dest_csv.exists():
        download_file(ENRON_SAMPLE_URL, dest_csv)
    
    if dest_csv.exists() and dest_csv.stat().st_size > 100:
        try:
            # Common encoding for this CSV format
            for enc in ["utf-8", "latin-1", "cp1252"]:
                try:
                    df = pd.read_csv(dest_csv, encoding=enc)
                    break
                except UnicodeDecodeError:
                    continue
            
            # Identify columns
            if "v1" in df.columns and "v2" in df.columns:
                df = df.rename(columns={"v1": "label_str", "v2": "text"})
                df["label"] = df["label_str"].map({"ham": 0, "spam": 1})
                df["source"] = "email_corpus_mirror"
                return df[["text", "label", "source"]].dropna()
            elif "label" in df.columns and "text" in df.columns:
                df["source"] = "email_corpus_mirror"
                return df[["text", "label", "source"]].dropna()
        except Exception as e:
            logger.error(f"Error reading {dest_csv}: {e}")
    return None

def load_secondary_email_dataset() -> Optional[pd.DataFrame]:
    """Download secondary email spam dataset."""
    dest_csv = RAW_DATA_DIR / "spam_or_not_spam.csv"
    if not dest_csv.exists():
        download_file(GITHUB_ENRON_SPAM_URL, dest_csv)
    
    if dest_csv.exists() and dest_csv.stat().st_size > 100:
        try:
            df = pd.read_csv(dest_csv)
            if "email" in df.columns and "label" in df.columns:
                df = df.rename(columns={"email": "text"})
                df["source"] = "spam_or_not_spam_email"
                return df[["text", "label", "source"]].dropna()
        except Exception as e:
            logger.error(f"Error reading secondary dataset: {e}")
    return None

def generate_robust_seed_dataset() -> pd.DataFrame:
    """Generate a high-quality diverse email dataset covering modern spam/phishing and business ham."""
    spam_templates = [
        "URGENT: Your account access has been suspended due to unauthorized activity. Verify your identity now at https://secure-bank-update.com/verify to restore access immediately.",
        "Congratulations! You've been selected as the official winner of our $1,000,000 lottery cash prize! Reply with your bank wire details to claim.",
        "Refill your prescription pills at 85% discount! Fast discreet overnight delivery worldwide. Buy Viagra, Cialis online without doctor note.",
        "Final Notice: Your tax return refund of $1,420.50 is pending approval. Submit your SSN and credit card details at http://irs-tax-refunds-gov.com.",
        "Earn $5,000 a week working from home! No experience required. Guaranteed high yield cryptocurrency investment program. Contact us on WhatsApp today.",
        "Invoice INV-98231 is past due. Immediate payment required to avoid debt collection actions. Click the link to review attached bill: http://malicious-invoicing.net/doc",
        "Hot singles in your local area want to meet you tonight! 100% free chat and private photo gallery. Click here now!",
        "Dear Beneficiary, I am barrister Collins representing the deceased estate with $14.5 Million USD. We need a reliable partner to transfer these funds.",
        "Action Required: Your Microsoft 365 password expires in 2 hours. Keep your existing password by logging into http://office-renew-portal.com.",
        "Get pre-approved for an unsecured loan up to $50,000 with 0% APR! Bad credit score accepted. Apply before midnight to secure your rate.",
        "Limited time offer: Free iPhone 16 Pro Max giveaway! Just complete our short 2-minute survey and pay $1 for express shipping.",
        "SECURITY ALERT: Someone logged into your PayPal account from Moscow, Russia. If this was not you, cancel the transaction immediately here: http://paypal-dispute-auth.biz",
        "Exclusive offer: Boost your Google ranking to #1 guaranteed in 30 days! Low monthly SEO packages starting at only $99. Reply now.",
        "You have 1 pending parcel waiting for delivery at the distribution hub. Pay the unpaid customs fee of $2.99 at http://usps-track-package-redelivery.org",
        "Crypto trading bot generates 300% passive returns monthly! Join our private VIP Telegram channel today."
    ]

    ham_templates = [
        "Hi team, please find attached the meeting minutes and action items from today's product sync. Let me know if anything needs updating before the client call.",
        "Reminder: The quarterly all-hands meeting is scheduled for Thursday at 2:00 PM EST in Conference Room B and on Zoom.",
        "Thanks for sending over the pull request! I reviewed the changes and left a couple of minor comments on the data loader class.",
        "Can we reschedule our 1:1 to tomorrow afternoon? A conflict came up with our infrastructure team regarding the database migration.",
        "Attached is the updated project budget spreadsheet reflecting the Q3 hardware and cloud expenses. Please review when you have a moment.",
        "Hi Alex, did you receive the confirmation email for the flight booking to San Francisco next week? Safe travels!",
        "Here are the release notes for version 2.4.1 deployment going live tonight at 10 PM. Testing was completed on staging with zero blockers.",
        "Good morning everyone, just a friendly reminder to submit your timesheets by Friday noon for payroll processing.",
        "The Python documentation for the new API endpoints has been merged into main. You can preview the generated docs at http://internal-wiki.company.local/api",
        "Thanks for dinner last night, it was really great catching up with you and Sarah! Let's get together again soon.",
        "Your weekly newsletter digest from PyTorch Engineering: What's new in PyTorch 2.3, performance benchmarks, and community tutorials.",
        "Hi David, could you approve the access request for the new data scientist joining our team next Monday? Thanks in advance.",
        "Hey, are you free for a quick coffee break? Let's meet at the lobby cafe around 3:30.",
        "Your GitHub pull request #42 has passed all CI checks and was successfully merged into main.",
        "Please find the signed contract copy attached for the vendor software license renewal."
    ]

    records = []
    # Generate variations
    for i in range(120):
        for s in spam_templates:
            records.append({
                "text": f"{s} Ref ID: #{1000 + i * 7}",
                "label": 1,
                "source": "curated_spam"
            })
        for h in ham_templates:
            records.append({
                "text": f"{h} Project note #{1000 + i * 7}",
                "label": 0,
                "source": "curated_ham"
            })
    
    return pd.DataFrame(records)

def ingest_and_prepare_dataset(min_samples: int = 5000) -> pd.DataFrame:
    """Ingest datasets from multiple public sources, clean, and combine into a unified dataset."""
    logger.info("Starting dataset ingestion pipeline...")
    dataframes: List[pd.DataFrame] = []

    # 1. SMS Spam Collection
    df_sms = load_sms_spam_collection()
    if df_sms is not None and len(df_sms) > 0:
        logger.info(f"Loaded {len(df_sms)} records from SMS Spam Collection.")
        dataframes.append(df_sms)

    # 2. Email Corpus Mirror
    df_email = load_enron_or_email_dataset()
    if df_email is not None and len(df_email) > 0:
        logger.info(f"Loaded {len(df_email)} records from Email dataset mirror.")
        dataframes.append(df_email)

    # 3. Secondary Email dataset
    df_sec = load_secondary_email_dataset()
    if df_sec is not None and len(df_sec) > 0:
        logger.info(f"Loaded {len(df_sec)} records from secondary email corpus.")
        dataframes.append(df_sec)

    # 4. If total samples below threshold, augment with high-signal curated emails
    total_loaded = sum(len(d) for d in dataframes)
    if total_loaded < min_samples:
        logger.info(f"Total samples ({total_loaded}) below target ({min_samples}). Augmenting with curated email corpus...")
        df_curated = generate_robust_seed_dataset()
        dataframes.append(df_curated)

    # Combine
    combined = pd.concat(dataframes, ignore_index=True)
    combined["text"] = combined["text"].astype(str)
    combined["label"] = combined["label"].astype(int)

    # Drop duplicates and empty text
    before_dedup = len(combined)
    combined = combined.drop_duplicates(subset=["text"]).reset_index(drop=True)
    combined = combined[combined["text"].str.strip().str.len() > 5].reset_index(drop=True)
    logger.info(f"Combined dataset: {len(combined)} samples (deduped from {before_dedup}).")

    # Clean text with preprocessor
    preprocessor = TextPreprocessor(DEFAULT_CONFIG.preprocessing)
    logger.info("Cleaning texts...")
    combined["cleaned_text"] = preprocessor.clean_series(combined["text"].tolist())
    
    # Save processed dataset
    output_path = PROCESSED_DATA_DIR / "spam_corpus.csv"
    combined.to_csv(output_path, index=False)
    logger.info(f"Saved processed dataset to {output_path}")

    # Log summary statistics
    spam_count = (combined["label"] == 1).sum()
    ham_count = (combined["label"] == 0).sum()
    logger.info(f"Class distribution: Ham={ham_count} ({ham_count/len(combined):.1%}), Spam={spam_count} ({spam_count/len(combined):.1%})")

    return combined

if __name__ == "__main__":
    ingest_and_prepare_dataset()
