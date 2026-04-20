from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

PERSONAS = {
    "Bot A": "Tech Maximalist: I believe AI and crypto will solve all human problems. I am highly optimistic about technology, Elon Musk, and space exploration. I dismiss regulatory concerns.",
    "Bot B": "Doomer / Skeptic: I believe late-stage capitalism and tech monopolies are destroying society. I am highly critical of AI, social media, and billionaires. I value privacy and nature.",
    "Bot C": "Finance Bro: I strictly care about markets, interest rates, trading algorithms, and making money. I speak in finance jargon and view everything through the lens of ROI."
}

class PersonaRouter:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = None
        self.init_db()

    def init_db(self):
        docs = [
            Document(page_content=content, metadata={"bot_name": name})
            for name, content in PERSONAS.items()
        ]
        self.vectorstore = FAISS.from_documents(docs, self.embeddings)

    def route_post_to_bots(self, post_content: str, threshold: float = 0.40):
        results = self.vectorstore.similarity_search_with_score(post_content, k=3)
        
        matches = []
        for doc, score in results:
            sim = 1 - (score / 2.0)
            if sim > threshold:
                matches.append({
                    "bot_name": doc.metadata["bot_name"],
                    "similarity": round(sim, 4),
                    "persona": doc.page_content
                })
                
        return matches

if __name__ == "__main__":
    router = PersonaRouter()
    post = "OpenAI just released a new model that might replace junior developers."
    bots = router.route_post_to_bots(post, threshold=0.1) 
    print(bots)
