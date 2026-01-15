"""
Build FAQ Embeddings Index
==========================

Run this script once before starting the bot to build the embeddings index.
This takes ~30 seconds on first run, then the bot loads instantly.

Usage:
    python build_index.py
"""

import logging
from pathlib import Path

from faq_embeddings_db import FAQEmbeddingsDB
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

def main():
    print("=" * 50)
    print("Building FAQ Embeddings Index")
    print("=" * 50)
    
    faq_path = Path(config.FAQ_JSON_PATH)
    if not faq_path.exists():
        print(f"\n❌ Error: FAQ file not found at {faq_path}")
        print("Make sure data/faq.json exists!")
        return
    
    print(f"\n📂 Loading FAQ from: {faq_path}")
    db = FAQEmbeddingsDB(config.FAQ_JSON_PATH)
    print(f"✓ Loaded {len(db.items)} FAQ items")
    
    print("\n🔄 Building embeddings (this may take a minute)...")
    db.build_index()
    
    print(f"\n💾 Saving index to: {config.FAQ_INDEX_PATH}")
    db.save(config.FAQ_INDEX_PATH)
    
    print("\n" + "=" * 50)
    print("✅ Index built successfully!")
    print("=" * 50)
    
    print("\n🧪 Testing search...")
    test_queries = [
        "когда сессия?",
        "как получить стипендию",
        "где находится общежитие"
    ]
    
    for query in test_queries:
        results = db.search(query, top_k=1)
        if results:
            print(f"\n  Q: {query}")
            print(f"  → [{results[0].score:.2f}] {results[0].item.question}")
    
    print("\n✨ Ready! Now run: python bot.py")


if __name__ == "__main__":
    main()
