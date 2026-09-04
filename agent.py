# # (venv) PS D:\Agentic AI\AI-powered bank customer-support assistant\AI-powered-bank-customer-support-assistant> python main.py
# # warning: The `fitz` API is deprecated and will be removed in future. Use `import pymupdf` instead.
# # USER_AGENT environment variable not set, consider setting it to identify your requests.

# # ============================================================
# # STEP 1 — LOADING DOCUMENTS
# # ============================================================
# # Normal PDF detected: account_opening_policy.pdf
# # Loaded 2 pages from account_opening_policy.pdf
# # Normal PDF detected: complaint_handling_policy.pdf
# # Loaded 3 pages from complaint_handling_policy.pdf
# # Normal PDF detected: credit_card_policy.pdf
# # Loaded 3 pages from credit_card_policy.pdf
# # Normal PDF detected: customer_faq.pdf
# # Loaded 3 pages from customer_faq.pdf
# # Normal PDF detected: loan_policy.pdf
# # Loaded 3 pages from loan_policy.pdf
# # Normal PDF detected: schedule_of_charges.pdf
# # Loaded 2 pages from schedule_of_charges.pdf
# # Scanned PDF detected: spytm_Scanned.pdf
# # Running OCR on: spytm_Scanned.pdf
# #   OCR page 1 of 1

# # Total documents loaded: 17
# # Total documents loaded: 17

# # ============================================================
# # STEP 2 — CLEANING DOCUMENTS
# # ============================================================
# # Cleaned 17 documents
# # Removed 0 documents that were too short after cleaning
# # Total documents after cleaning: 17

# # ============================================================
# # STEP 4 — INCREMENTAL INDEXING
# # ============================================================

# # Incremental indexing summary:
# #   New documents:     0
# #   Updated documents: 0
# #   Skipped (unchanged): 7
# #   Skipped: ['account_opening_policy.pdf', 'complaint_handling_policy.pdf', 'credit_card_policy.pdf', 'customer_faq.pdf', 'loan_policy.pdf', 'schedule_of_charges.pdf', 'spytm_Scanned.pdf']

# # All documents already indexed — loading existing store
# # D:\Agentic AI\AI-powered bank customer-support assistant\AI-powered-bank-customer-support-assistant\retrieval\embedder.py:15: LangChainDeprecationWarning: The class `HuggingFaceEmbeddings` was deprecated in LangChain 0.2.2 and will be removed in 1.0. An updated version of the class exists in the `langchain-huggingface package and should be used instead. To use it run `pip install -U `langchain-huggingface` and import as `from `langchain_huggingface import HuggingFaceEmbeddings``.
# #   return HuggingFaceEmbeddings(
# # Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
# # Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 103/103 [00:00<00:00, 5722.11it/s]
# # D:\Agentic AI\AI-powered bank customer-support assistant\AI-powered-bank-customer-support-assistant\retrieval\store.py:17: LangChainDeprecationWarning: The class `Chroma` was deprecated in LangChain 0.2.9 and will be removed in 1.0. An updated version of the class exists in the `langchain-chroma package and should be used instead. To use it run `pip install -U `langchain-chroma` and import as `from `langchain_chroma import Chroma``.
# #   return Chroma(
# # Total chunks in vector store: 66
# # Vector store ready
# # account_opening_policy.pdf → policy
# # complaint_handling_policy.pdf → policy
# # credit_card_policy.pdf → policy
# # customer_faq.pdf → faq
# # loan_policy.pdf → policy
# # schedule_of_charges.pdf → charges
# # spytm_Scanned.pdf → general
# # Recommended for policy: {'chunk_size': 600, 'overlap': 100}
# # Recursive chunking: 2 docs → 7 chunks
# # Recommended for policy: {'chunk_size': 600, 'overlap': 100}
# # Recursive chunking: 3 docs → 11 chunks
# # Recommended for policy: {'chunk_size': 600, 'overlap': 100}
# # Recursive chunking: 3 docs → 8 chunks
# # Recommended for faq: {'chunk_size': 400, 'overlap': 50}
# # Recursive chunking: 3 docs → 18 chunks
# # Recommended for policy: {'chunk_size': 600, 'overlap': 100}
# # Recursive chunking: 3 docs → 8 chunks
# # Recommended for charges: {'chunk_size': 300, 'overlap': 30}
# # Recursive chunking: 2 docs → 13 chunks
# # Recommended for general: {'chunk_size': 500, 'overlap': 50}
# # Recursive chunking: 1 docs → 1 chunks

# # Chunks prepared for BM25: 66

# # ============================================================
# # STEP 6 — SETTING UP LLM
# # ============================================================
# # LLM configured

# # ============================================================
# # STEP 7 — CREATING RAG PROMPT
# # ============================================================
# # RAG prompt created

# # ============================================================
# # STEP 9 — BUILDING RAG CHAIN
# # ============================================================
# # RAG chain ready

# # ============================================================
# # DEBUGGING: What is the minimum balance required for a savings account?
# # ============================================================

# # --- STEP 1: RETRIEVAL ---

# # ==================================================
# # SMART RETRIEVAL
# # Strategy: ensemble | k=4 | reorder=True
# # Question: What is the minimum balance required for a savings account?...
# # ==================================================
# # Detected relevant document: account_opening_policy.pdf (score: 2)

# # Reordered 4 chunks — best at start and end

# # Final context: 2493 chars from 4 chunks
# # Sources: ['account_opening_policy.pdf', 'account_opening_policy.pdf', 'account_opening_policy.pdf', 'account_opening_policy.pdf']

# # Retrieved 4 chunks:

# #   Chunk 1:
# #   Source: account_opening_policy.pdf
# #   Page: 1
# #   Preview: Foreign Currency Account: USD 100 minimum balance 
 
# # Below minimum balance charge: Rs. 500 per month

# #   Chunk 2:
# #   Source: account_opening_policy.pdf
# #   Page: 2
# #   Preview: Debit card issued within 7 working days 
# # Online banking activated within 24 hours 
# # Cheque book issue

# #   Chunk 3:
# #   Source: account_opening_policy.pdf
# #   Page: 1
# #   Preview: ABC BANK — ACCOUNT OPENING POLICY 
# # Version 1.0 | Effective Date: January 2024 
 
# # SECTION 1 — TYPES O

# #   Chunk 4:
# #   Source: account_opening_policy.pdf
# #   Page: 1
# #   Preview: 1.2 Business Accounts 
# # Current Account for Business — unlimited transactions, no profit 
# # Business Sa

# # --- STEP 2: AUGMENTATION ---
# # Context length: 2493 characters
# # Context preview:
# # [Document: account_opening_policy.pdf | Page: 1]
# # Foreign Currency Account: USD 100 minimum balance 
 
# # Below minimum balance charge: Rs. 500 per month for savings 
# # Below minimum balance charge: Rs. 1,000 per month for current 
 
# # SECTION 3 — PROFIT RATES 
 
# # Regular Savings Account: 5% per annum 
# # Premium Savings Account: 7% per annum 
# # Basic Banking Account: 0% profit 
# # Profit credited: monthly on last...

# # --- STEP 3: GENERATION ---

# # ==================================================
# # SMART RETRIEVAL
# # Strategy: ensemble | k=4 | reorder=True
# # Question: What is the minimum balance required for a savings account?...
# # ==================================================
# # Detected relevant document: account_opening_policy.pdf (score: 2)

# # Reordered 4 chunks — best at start and end

# # Final context: 2493 chars from 4 chunks
# # Sources: ['account_opening_policy.pdf', 'account_opening_policy.pdf', 'account_opening_policy.pdf', 'account_opening_policy.pdf']

# # Answer:
# # The minimum balance required for a Regular Savings Account is **Rs. 10,000**.

# # ============================================================
# # DEBUGGING: Which documents do I need to open a bank account?
# # ============================================================

# # --- STEP 1: RETRIEVAL ---

# # ==================================================
# # SMART RETRIEVAL
# # Strategy: ensemble | k=4 | reorder=True
# # Question: Which documents do I need to open a bank account?...
# # ==================================================
# # Detected relevant document: account_opening_policy.pdf (score: 2)

# # Reordered 4 chunks — best at start and end

# # Final context: 2132 chars from 4 chunks
# # Sources: ['account_opening_policy.pdf', 'account_opening_policy.pdf', 'account_opening_policy.pdf', 'account_opening_policy.pdf']

# # Retrieved 4 chunks:

# #   Chunk 1:
# #   Source: account_opening_policy.pdf
# #   Page: 1
# #   Preview: 1.2 Business Accounts 
# # Current Account for Business — unlimited transactions, no profit 
# # Business Sa

# #   Chunk 2:
# #   Source: account_opening_policy.pdf
# #   Page: 2
# #   Preview: Minor original B-Form 
# # Parent or guardian original CNIC 
# # Proof of relationship — birth certificate 


# #   Chunk 3:
# #   Source: account_opening_policy.pdf
# #   Page: 1
# #   Preview: Original CNIC — mandatory 
# # Utility bill not older than 3 months — proof of address 
# # Passport size ph

# #   Chunk 4:
# #   Source: account_opening_policy.pdf
# #   Page: 1
# #   Preview: ABC BANK — ACCOUNT OPENING POLICY 
# # Version 1.0 | Effective Date: January 2024 
 
# # SECTION 1 — TYPES O

# # --- STEP 2: AUGMENTATION ---
# # Context length: 2132 characters
# # Context preview:
# # [Document: account_opening_policy.pdf | Page: 1]
# # 1.2 Business Accounts 
# # Current Account for Business — unlimited transactions, no profit 
# # Business Savings Account — limited transactions with profit 
# # Corporate Account — for registered companies 
 
# # SECTION 2 — MINIMUM BALANCE REQUIREMENTS 
 
# # Basic Banking Account: Rs. 0 minimum balance 
# # Regular Savings Account: Rs. 10,000 minimum balance 
# # Premium Sa...

# # --- STEP 3: GENERATION ---

# # ==================================================
# # SMART RETRIEVAL
# # Strategy: ensemble | k=4 | reorder=True
# # Question: Which documents do I need to open a bank account?...
# # ==================================================
# # Detected relevant document: account_opening_policy.pdf (score: 2)

# # Reordered 4 chunks — best at start and end

# # Final context: 2132 chars from 4 chunks
# # Sources: ['account_opening_policy.pdf', 'account_opening_policy.pdf', 'account_opening_policy.pdf', 'account_opening_policy.pdf']

# # Answer:
# # To open an account at ABC Bank you’ll need to provide the following documents:

# # | Account type | Required documents |
# # |--------------|--------------------|
# # | **All accounts (individual, savings, current, foreign‑currency, etc.)** | • Original CNIC (mandatory)<br>• Utility bill (not older than 3 months) – proof of address<br>• 2 passport‑size photographs<br>• Source of income declaration form (mandatory)<br>• Next‑of‑kin details (mandatory) |
# # | **Minor accounts** | • Minor’s original B‑Form<br>• Parent or guardian’s original CNIC<br>• Proof of relationship (birth certificate)<br>• Guardian’s utility bill |
# # | **Business accounts** | • Business owner’s CNIC<br>• Business registration certificate<br>• NTN certificate<br>• Board resolution for authorized signatories<br>• Partnership deed (if applicable)<br>• Memorandum and articles of association (for companies) |

# # Make sure all documents are original and up‑to‑date. Once you submit them, the account will be activated within 24 hours and a debit card will be issued within 7 working days.

# # ============================================================
# # DEBUGGING: How much is the IBFT transfer charge?
# # ============================================================

# # --- STEP 1: RETRIEVAL ---

# # ==================================================
# # SMART RETRIEVAL
# # Strategy: ensemble | k=4 | reorder=True
# # Question: How much is the IBFT transfer charge?...
# # ==================================================
# # Detected relevant document: schedule_of_charges.pdf (score: 3)

# # Reordered 4 chunks — best at start and end

# # Final context: 1314 chars from 4 chunks
# # Sources: ['schedule_of_charges.pdf', 'schedule_of_charges.pdf', 'schedule_of_charges.pdf', 'schedule_of_charges.pdf']

# # Retrieved 4 chunks:

# #   Chunk 1:
# #   Source: schedule_of_charges.pdf
# #   Page: 1
# #   Preview: Cheque return charges inward: Rs. 1,000 per cheque 
# # Cheque return charges outward: Rs. 500 per chequ

# #   Chunk 2:
# #   Source: schedule_of_charges.pdf
# #   Page: 2
# #   Preview: Foreign currency conversion rate: SBP rate plus 3.5% 
 
# # SECTION 4 — CARD CHARGES 
 
# # Debit card issua

# #   Chunk 3:
# #   Source: schedule_of_charges.pdf
# #   Page: 2
# #   Preview: Profit and loss certificate: Rs. 500 
# # Reference letter: Rs. 1,000 
# # Swift message charges outward: Rs

# #   Chunk 4:
# #   Source: schedule_of_charges.pdf
# #   Page: 1
# #   Preview: IBFT above Rs. 500,000: Rs. 500 per transaction 
# # Internal fund transfer within ABC Bank: Free unlimi

# # --- STEP 2: AUGMENTATION ---
# # Context length: 1314 characters
# # Context preview:
# # [Document: schedule_of_charges.pdf | Page: 1]
# # Cheque return charges inward: Rs. 1,000 per cheque 
# # Cheque return charges outward: Rs. 500 per cheque 
# # Stop payment instruction: Rs. 500 
 
# # 2.3 Online and Digital Transactions 
# # IBFT up to Rs. 25,000: Rs. 200 per transaction 
# # IBFT Rs. 25,001 to Rs. 500,000: Rs. 300 per transaction

# # ---

# # [Document: schedule_of_charges.pdf | Page: 2]
# # Foreign currency conv...

# # --- STEP 3: GENERATION ---

# # ==================================================
# # SMART RETRIEVAL
# # Strategy: ensemble | k=4 | reorder=True
# # Question: How much is the IBFT transfer charge?...
# # ==================================================
# # Detected relevant document: schedule_of_charges.pdf (score: 3)

# # Reordered 4 chunks — best at start and end

# # Final context: 1314 chars from 4 chunks
# # Sources: ['schedule_of_charges.pdf', 'schedule_of_charges.pdf', 'schedule_of_charges.pdf', 'schedule_of_charges.pdf']

# # Answer:
# # IBFT transfer charges vary by the amount transferred:

# # | Transfer amount | Charge per transaction |
# # |-----------------|------------------------|
# # | Up to Rs. 25,000 | Rs. 200 |
# # | Rs. 25,001 – Rs. 500,000 | Rs. 300 |
# # | Above Rs. 500,000 | Rs. 500 |

# # ============================================================
# # DEBUGGING: What is the procedure for filing a complaint against the bank?
# # ============================================================

# # --- STEP 1: RETRIEVAL ---

# # ==================================================
# # SMART RETRIEVAL
# # Strategy: ensemble | k=4 | reorder=True
# # Question: What is the procedure for filing a complaint against the ban...
# # ==================================================
# # Detected relevant document: complaint_handling_policy.pdf (score: 1)

# # Reordered 4 chunks — best at start and end

# # Final context: 2051 chars from 4 chunks
# # Sources: ['complaint_handling_policy.pdf', 'complaint_handling_policy.pdf', 'complaint_handling_policy.pdf', 'complaint_handling_policy.pdf']

# # Retrieved 4 chunks:

# #   Chunk 1:
# #   Source: complaint_handling_policy.pdf
# #   Page: 3
# #   Preview: Right to escalate if not satisfied with resolution 
# # Right to approach Banking Mohtasib if internal r

# #   Chunk 2:
# #   Source: complaint_handling_policy.pdf
# #   Page: 1
# #   Preview: CNIC number 
# # Account number or card number 
# # Contact number and email address 
# # Clear description of t

# #   Chunk 3:
# #   Source: complaint_handling_policy.pdf
# #   Page: 1
# #   Preview: ABC BANK — COMPLAINT HANDLING POLICY 
# # Version 1.0 | Effective Date: January 2024 
 
# # SECTION 1 — POLI

# #   Chunk 4:
# #   Source: complaint_handling_policy.pdf
# #   Page: 1
# #   Preview: 2.1 Available Channels 
# # Phone: Call 0800-ABCBANK available 24 hours 7 days a week 
# # Branch: Visit any

# # --- STEP 2: AUGMENTATION ---
# # Context length: 2051 characters
# # Context preview:
# # [Document: complaint_handling_policy.pdf | Page: 3]
# # Right to escalate if not satisfied with resolution 
# # Right to approach Banking Mohtasib if internal resolution fails 
# # Right to confidentiality of complaint details

# # ---

# # [Document: complaint_handling_policy.pdf | Page: 1]
# # CNIC number 
# # Account number or card number 
# # Contact number and email address 
# # Clear description of the complaint 
# # Date and loca...

# # --- STEP 3: GENERATION ---

# # ==================================================
# # SMART RETRIEVAL
# # Strategy: ensemble | k=4 | reorder=True
# # Question: What is the procedure for filing a complaint against the ban...
# # ==================================================
# # Detected relevant document: complaint_handling_policy.pdf (score: 1)

# # Reordered 4 chunks — best at start and end

# # Final context: 2051 chars from 4 chunks
# # Sources: ['complaint_handling_policy.pdf', 'complaint_handling_policy.pdf', 'complaint_handling_policy.pdf', 'complaint_handling_policy.pdf']

# # Answer:
# # **Procedure for filing a complaint against ABC Bank**

# # 1. **Choose a filing channel**  
# #    - **Phone**: Call 0800‑ABCBANK (available 24 h/7 d).  
# #    - **Branch**: Visit any ABC Bank branch and speak to a Customer Service Officer.  
# #    - **Email**: Send an email to complaints@abcbank.com (response within 24 h).  
# #    - **Mobile App**: Open the ABC Bank app → Help section → File Complaint.  
# #    - **Website**: Go to abcbank.com/complaints and fill out the online form.  
# #    - **Written**: Mail a letter to the Complaints Department, Head Office, Karachi.

# # 2. **Provide the required information**  
# #    - Full name as per CNIC  
# #    - CNIC number  
# #    - Account number or card number  
# #    - Contact number and email address  
# #    - Clear description of the complaint  
# #    - Date and location of the incident  
# #    - Any reference numbers related to the issue  
# #    - Supporting documents (if available)

# # 3. **After registration**  
# #    - A complaint reference number is issued immediately.  
# #    - You will receive an SMS confirmation within 1 hour and an email confirmation within 2 hours.  
# #    - Use this reference number for all follow‑up inquiries.

# # Follow these steps to file a complaint and ensure it is processed efficiently.

# # ============================================================
# # DEBUGGING: How much does a RAAST transfer cost?
# # ============================================================

# # --- STEP 1: RETRIEVAL ---

# # ==================================================
# # SMART RETRIEVAL
# # Strategy: ensemble | k=4 | reorder=True
# # Question: How much does a RAAST transfer cost?...
# # ==================================================
# # Detected relevant document: schedule_of_charges.pdf (score: 3)

# # Reordered 4 chunks — best at start and end

# # Final context: 1220 chars from 4 chunks
# # Sources: ['schedule_of_charges.pdf', 'schedule_of_charges.pdf', 'schedule_of_charges.pdf', 'schedule_of_charges.pdf']

# # Retrieved 4 chunks:

# #   Chunk 1:
# #   Source: schedule_of_charges.pdf
# #   Page: 1
# #   Preview: Mini statement at own ATM: Free up to 2 per month 
# # Mini statement above 2: Rs. 25 per transaction 
 

# #   Chunk 2:
# #   Source: schedule_of_charges.pdf
# #   Page: 2
# #   Preview: Profit and loss certificate: Rs. 500 
# # Reference letter: Rs. 1,000 
# # Swift message charges outward: Rs

# #   Chunk 3:
# #   Source: schedule_of_charges.pdf
# #   Page: 2
# #   Preview: Balance inquiry at other bank ATM: Rs. 18 per transaction 
# # Maximum 5 free transactions per month for

# #   Chunk 4:
# #   Source: schedule_of_charges.pdf
# #   Page: 1
# #   Preview: IBFT above Rs. 500,000: Rs. 500 per transaction 
# # Internal fund transfer within ABC Bank: Free unlimi

# # --- STEP 2: AUGMENTATION ---
# # Context length: 1220 characters
# # Context preview:
# # [Document: schedule_of_charges.pdf | Page: 1]
# # Mini statement at own ATM: Free up to 2 per month 
# # Mini statement above 2: Rs. 25 per transaction 
 
# # 3.2 Other Bank ATM in Pakistan 
# # Cash withdrawal at other bank ATM: Rs. 25 per transaction

# # ---

# # [Document: schedule_of_charges.pdf | Page: 2]
# # Profit and loss certificate: Rs. 500 
# # Reference letter: Rs. 1,000 
# # Swift message charges outward: Rs. 1,500 
 
# # ...

# # --- STEP 3: GENERATION ---

# # ==================================================
# # SMART RETRIEVAL
# # Strategy: ensemble | k=4 | reorder=True
# # Question: How much does a RAAST transfer cost?...
# # ==================================================
# # Detected relevant document: schedule_of_charges.pdf (score: 3)

# # Reordered 4 chunks — best at start and end

# # Final context: 1220 chars from 4 chunks
# # Sources: ['schedule_of_charges.pdf', 'schedule_of_charges.pdf', 'schedule_of_charges.pdf', 'schedule_of_charges.pdf']

# # Answer:
# # RAAST transfer: Free unlimited.
# # (venv) PS D:\Agentic AI\AI-powered bank customer-support assistant\AI-powered-bank-customer-support-assistant> 


# # (venv) PS D:\Agentic AI\AI-powered bank customer-support assistant\AI-powered-bank-customer-support-assistant> python main.py
# # warning: The `fitz` API is deprecated and will be removed in future. Use `import pymupdf` instead.
# # USER_AGENT environment variable not set, consider setting it to identify your requests.

# # ============================================================
# # STEP 1 — LOADING DOCUMENTS
# # ============================================================
# # Normal PDF detected: account_opening_policy.pdf
# # Loaded 2 pages from account_opening_policy.pdf
# # Normal PDF detected: complaint_handling_policy.pdf
# # Loaded 3 pages from complaint_handling_policy.pdf
# # Normal PDF detected: credit_card_policy.pdf
# # Loaded 3 pages from credit_card_policy.pdf
# # Normal PDF detected: customer_faq.pdf
# # Loaded 3 pages from customer_faq.pdf
# # Normal PDF detected: loan_policy.pdf
# # Loaded 3 pages from loan_policy.pdf
# # Normal PDF detected: schedule_of_charges.pdf
# # Loaded 2 pages from schedule_of_charges.pdf
# # Scanned PDF detected: spytm_Scanned.pdf
# # Running OCR on: spytm_Scanned.pdf
# #   OCR page 1 of 1

# # Total documents loaded: 17
# # Total documents loaded: 17

# # ============================================================
# # STEP 2 — CLEANING DOCUMENTS
# # ============================================================
# # Cleaned 17 documents
# # Removed 0 documents that were too short after cleaning
# # Total documents after cleaning: 17

# # ============================================================
# # STEP 4 — INCREMENTAL INDEXING
# # ============================================================

# # Incremental indexing summary:
# #   New documents:     0
# #   Updated documents: 0
# #   Skipped (unchanged): 7
# #   Skipped: ['account_opening_policy.pdf', 'complaint_handling_policy.pdf', 'credit_card_policy.pdf', 'customer_faq.pdf', 'loan_policy.pdf', 'schedule_of_charges.pdf', 'spytm_Scanned.pdf']

# # All documents already indexed — loading existing store
# # D:\Agentic AI\AI-powered bank customer-support assistant\AI-powered-bank-customer-support-assistant\retrieval\embedder.py:15: LangChainDeprecationWarning: The class `HuggingFaceEmbeddings` was deprecated in LangChain 0.2.2 and will be removed in 1.0. An updated version of the class exists in the `langchain-huggingface package and should be used instead. To use it run `pip install -U `langchain-huggingface` and import as `from `langchain_huggingface import HuggingFaceEmbeddings``.
# #   return HuggingFaceEmbeddings(
# # Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
# # Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 103/103 [00:00<00:00, 2577.65it/s]
# # D:\Agentic AI\AI-powered bank customer-support assistant\AI-powered-bank-customer-support-assistant\retrieval\store.py:17: LangChainDeprecationWarning: The class `Chroma` was deprecated in LangChain 0.2.9 and will be removed in 1.0. An updated version of the class exists in the `langchain-chroma package and should be used instead. To use it run `pip install -U `langchain-chroma` and import as `from `langchain_chroma import Chroma``.
# #   return Chroma(
# # Total chunks in vector store: 66
# # Vector store ready
# # account_opening_policy.pdf → policy
# # complaint_handling_policy.pdf → policy
# # credit_card_policy.pdf → policy
# # customer_faq.pdf → faq
# # loan_policy.pdf → policy
# # schedule_of_charges.pdf → charges
# # spytm_Scanned.pdf → general
# # Recommended for policy: {'chunk_size': 600, 'overlap': 100}
# # Recursive chunking: 2 docs → 7 chunks
# # Recommended for policy: {'chunk_size': 600, 'overlap': 100}
# # Recursive chunking: 3 docs → 11 chunks
# # Recommended for policy: {'chunk_size': 600, 'overlap': 100}
# # Recursive chunking: 3 docs → 8 chunks
# # Recommended for faq: {'chunk_size': 400, 'overlap': 50}
# # Recursive chunking: 3 docs → 18 chunks
# # Recommended for policy: {'chunk_size': 600, 'overlap': 100}
# # Recursive chunking: 3 docs → 8 chunks
# # Recommended for charges: {'chunk_size': 300, 'overlap': 30}
# # Recursive chunking: 2 docs → 13 chunks
# # Recommended for general: {'chunk_size': 500, 'overlap': 50}
# # Recursive chunking: 1 docs → 1 chunks

# # Chunks prepared for BM25: 66

# # ============================================================
# # STEP 6 — SETTING UP LLM
# # ============================================================
# # LLM configured

# # ============================================================
# # STEP 7 — CREATING RAG PROMPT
# # ============================================================
# # RAG prompt created

# # ============================================================
# # STEP 9 — BUILDING RAG CHAIN
# # ============================================================
# # RAG chain ready

# # ============================================================
# # DEBUGGING: What is the minimum balance required for a savings account?
# # ============================================================

# # --- STEP 1: RETRIEVAL ---

# # ==================================================
# # SMART RETRIEVAL
# # Strategy: ensemble | k=4 | reorder=True
# # Question: What is the minimum balance required for a savings account?...
# # ==================================================
# # Detected relevant document: account_opening_policy.pdf (score: 2)

# # Reordered 4 chunks — best at start and end

# # Final context: 2493 chars from 4 chunks
# # Sources: ['account_opening_policy.pdf', 'account_opening_policy.pdf', 'account_opening_policy.pdf', 'account_opening_policy.pdf']

# # Retrieved 4 chunks:

# #   Chunk 1:
# #   Source: account_opening_policy.pdf
# #   Page: 1
# #   Preview: Foreign Currency Account: USD 100 minimum balance 
 
# # Below minimum balance charge: Rs. 500 per month

# #   Chunk 2:
# #   Source: account_opening_policy.pdf
# #   Page: 2
# #   Preview: Debit card issued within 7 working days 
# # Online banking activated within 24 hours 
# # Cheque book issue

# #   Chunk 3:
# #   Source: account_opening_policy.pdf
# #   Page: 1
# #   Preview: ABC BANK — ACCOUNT OPENING POLICY 
# # Version 1.0 | Effective Date: January 2024 
 
# # SECTION 1 — TYPES O

# #   Chunk 4:
# #   Source: account_opening_policy.pdf
# #   Page: 1
# #   Preview: 1.2 Business Accounts 
# # Current Account for Business — unlimited transactions, no profit 
# # Business Sa

# # --- STEP 2: AUGMENTATION ---
# # Context length: 2493 characters
# # Context preview:
# # [Document: account_opening_policy.pdf | Page: 1]
# # Foreign Currency Account: USD 100 minimum balance 
 
# # Below minimum balance charge: Rs. 500 per month for savings 
# # Below minimum balance charge: Rs. 1,000 per month for current 
 
# # SECTION 3 — PROFIT RATES 
 
# # Regular Savings Account: 5% per annum 
# # Premium Savings Account: 7% per annum 
# # Basic Banking Account: 0% profit 
# # Profit credited: monthly on last...

# # --- STEP 3: GENERATION ---

# # ==================================================
# # SMART RETRIEVAL
# # Strategy: ensemble | k=4 | reorder=True
# # Question: What is the minimum balance required for a savings account?...
# # ==================================================
# # Detected relevant document: account_opening_policy.pdf (score: 2)

# # Reordered 4 chunks — best at start and end

# # Final context: 2493 chars from 4 chunks
# # Sources: ['account_opening_policy.pdf', 'account_opening_policy.pdf', 'account_opening_policy.pdf', 'account_opening_policy.pdf']

# # Answer:
# # The minimum balance required for a Regular Savings Account is **Rs. 10,000**.

# # ============================================================
# # DEBUGGING: Which documents do I need to open a bank account?
# # ============================================================

# # --- STEP 1: RETRIEVAL ---

# # ==================================================
# # SMART RETRIEVAL
# # Strategy: ensemble | k=4 | reorder=True
# # Question: Which documents do I need to open a bank account?...
# # ==================================================
# # Detected relevant document: account_opening_policy.pdf (score: 2)

# # Reordered 4 chunks — best at start and end

# # Final context: 2132 chars from 4 chunks
# # Sources: ['account_opening_policy.pdf', 'account_opening_policy.pdf', 'account_opening_policy.pdf', 'account_opening_policy.pdf']

# # Retrieved 4 chunks:

# #   Chunk 1:
# #   Source: account_opening_policy.pdf
# #   Page: 1
# #   Preview: 1.2 Business Accounts 
# # Current Account for Business — unlimited transactions, no profit 
# # Business Sa

# #   Chunk 2:
# #   Source: account_opening_policy.pdf
# #   Page: 2
# #   Preview: Minor original B-Form 
# # Parent or guardian original CNIC 
# # Proof of relationship — birth certificate 


# #   Chunk 3:
# #   Source: account_opening_policy.pdf
# #   Page: 1
# #   Preview: Original CNIC — mandatory 
# # Utility bill not older than 3 months — proof of address 
# # Passport size ph

# #   Chunk 4:
# #   Source: account_opening_policy.pdf
# #   Page: 1
# #   Preview: ABC BANK — ACCOUNT OPENING POLICY 
# # Version 1.0 | Effective Date: January 2024 
 
# # SECTION 1 — TYPES O

# # --- STEP 2: AUGMENTATION ---
# # Context length: 2132 characters
# # Context preview:
# # [Document: account_opening_policy.pdf | Page: 1]
# # 1.2 Business Accounts 
# # Current Account for Business — unlimited transactions, no profit 
# # Business Savings Account — limited transactions with profit 
# # Corporate Account — for registered companies 
 
# # SECTION 2 — MINIMUM BALANCE REQUIREMENTS 
 
# # Basic Banking Account: Rs. 0 minimum balance 
# # Regular Savings Account: Rs. 10,000 minimum balance 
# # Premium Sa...

# # --- STEP 3: GENERATION ---

# # ==================================================
# # SMART RETRIEVAL
# # Strategy: ensemble | k=4 | reorder=True
# # Question: Which documents do I need to open a bank account?...
# # ==================================================
# # Detected relevant document: account_opening_policy.pdf (score: 2)

# # Reordered 4 chunks — best at start and end

# # Final context: 2132 chars from 4 chunks
# # Sources: ['account_opening_policy.pdf', 'account_opening_policy.pdf', 'account_opening_policy.pdf', 'account_opening_policy.pdf']

# # Answer:
# # To open an account at ABC Bank you’ll need to provide the following documents:

# # | Account type | Required documents |
# # |--------------|--------------------|
# # | **All accounts** | • Original CNIC (mandatory)  <br>• Utility bill (not older than 3 months) – proof of address  <br>• Two passport‑size photographs  <br>• Source of income declaration form (mandatory)  <br>• Next of kin details (mandatory) |
# # | **Minor accounts** | • Minor’s original B‑Form  <br>• Parent or guardian’s original CNIC  <br>• Proof of relationship (birth certificate)  <br>• Guardian’s utility bill |
# # | **Business accounts** | • Business owner’s CNIC  <br>• Business registration certificate  <br>• NTN certificate  <br>• Board resolution for authorized signatories  <br>• Partnership deed (if applicable)  <br>• Memorandum and articles of association (for companies) |

# # Make sure all documents are original and up‑to‑date before submitting them for account opening.

# # ============================================================
# # DEBUGGING: How much is the IBFT transfer charge?
# # ============================================================

# # --- STEP 1: RETRIEVAL ---

# # ==================================================
# # SMART RETRIEVAL
# # Strategy: ensemble | k=4 | reorder=True
# # Question: How much is the IBFT transfer charge?...
# # ==================================================
# # Detected relevant document: schedule_of_charges.pdf (score: 3)

# # Reordered 4 chunks — best at start and end

# # Final context: 1314 chars from 4 chunks
# # Sources: ['schedule_of_charges.pdf', 'schedule_of_charges.pdf', 'schedule_of_charges.pdf', 'schedule_of_charges.pdf']

# # Retrieved 4 chunks:

# #   Chunk 1:
# #   Source: schedule_of_charges.pdf
# #   Page: 1
# #   Preview: Cheque return charges inward: Rs. 1,000 per cheque 
# # Cheque return charges outward: Rs. 500 per chequ

# #   Chunk 2:
# #   Source: schedule_of_charges.pdf
# #   Page: 2
# #   Preview: Foreign currency conversion rate: SBP rate plus 3.5% 
 
# # SECTION 4 — CARD CHARGES 
 
# # Debit card issua

# #   Chunk 3:
# #   Source: schedule_of_charges.pdf
# #   Page: 2
# #   Preview: Profit and loss certificate: Rs. 500 
# # Reference letter: Rs. 1,000 
# # Swift message charges outward: Rs

# #   Chunk 4:
# #   Source: schedule_of_charges.pdf
# #   Page: 1
# #   Preview: IBFT above Rs. 500,000: Rs. 500 per transaction 
# # Internal fund transfer within ABC Bank: Free unlimi

# # --- STEP 2: AUGMENTATION ---
# # Context length: 1314 characters
# # Context preview:
# # [Document: schedule_of_charges.pdf | Page: 1]
# # Cheque return charges inward: Rs. 1,000 per cheque 
# # Cheque return charges outward: Rs. 500 per cheque 
# # Stop payment instruction: Rs. 500 
 
# # 2.3 Online and Digital Transactions 
# # IBFT up to Rs. 25,000: Rs. 200 per transaction 
# # IBFT Rs. 25,001 to Rs. 500,000: Rs. 300 per transaction

# # ---

# # [Document: schedule_of_charges.pdf | Page: 2]
# # Foreign currency conv...

# # --- STEP 3: GENERATION ---

# # ==================================================
# # SMART RETRIEVAL
# # Strategy: ensemble | k=4 | reorder=True
# # Question: How much is the IBFT transfer charge?...
# # ==================================================
# # Detected relevant document: schedule_of_charges.pdf (score: 3)

# # Reordered 4 chunks — best at start and end

# # Final context: 1314 chars from 4 chunks
# # Sources: ['schedule_of_charges.pdf', 'schedule_of_charges.pdf', 'schedule_of_charges.pdf', 'schedule_of_charges.pdf']

# # Answer:
# # IBFT transfer charges vary by the amount transferred:

# # | Transfer amount | Charge per transaction |
# # |-----------------|------------------------|
# # | Up to Rs. 25,000 | Rs. 200 |
# # | Rs. 25,001 – Rs. 500,000 | Rs. 300 |
# # | Above Rs. 500,000 | Rs. 500 |

# # ============================================================
# # DEBUGGING: What is the procedure for filing a complaint against the bank?
# # ============================================================

# # --- STEP 1: RETRIEVAL ---

# # ==================================================
# # SMART RETRIEVAL
# # Strategy: ensemble | k=4 | reorder=True
# # Question: What is the procedure for filing a complaint against the ban...
# # ==================================================
# # Detected relevant document: complaint_handling_policy.pdf (score: 1)

# # Reordered 4 chunks — best at start and end

# # Final context: 2051 chars from 4 chunks
# # Sources: ['complaint_handling_policy.pdf', 'complaint_handling_policy.pdf', 'complaint_handling_policy.pdf', 'complaint_handling_policy.pdf']

# # Retrieved 4 chunks:

# #   Chunk 1:
# #   Source: complaint_handling_policy.pdf
# #   Page: 3
# #   Preview: Right to escalate if not satisfied with resolution 
# # Right to approach Banking Mohtasib if internal r

# #   Chunk 2:
# #   Source: complaint_handling_policy.pdf
# #   Page: 1
# #   Preview: CNIC number 
# # Account number or card number 
# # Contact number and email address 
# # Clear description of t

# #   Chunk 3:
# #   Source: complaint_handling_policy.pdf
# #   Page: 1
# #   Preview: ABC BANK — COMPLAINT HANDLING POLICY 
# # Version 1.0 | Effective Date: January 2024 
 
# # SECTION 1 — POLI

# #   Chunk 4:
# #   Source: complaint_handling_policy.pdf
# #   Page: 1
# #   Preview: 2.1 Available Channels 
# # Phone: Call 0800-ABCBANK available 24 hours 7 days a week 
# # Branch: Visit any

# # --- STEP 2: AUGMENTATION ---
# # Context length: 2051 characters
# # Context preview:
# # [Document: complaint_handling_policy.pdf | Page: 3]
# # Right to escalate if not satisfied with resolution 
# # Right to approach Banking Mohtasib if internal resolution fails 
# # Right to confidentiality of complaint details

# # ---

# # [Document: complaint_handling_policy.pdf | Page: 1]
# # CNIC number 
# # Account number or card number 
# # Contact number and email address 
# # Clear description of the complaint 
# # Date and loca...

# # --- STEP 3: GENERATION ---

# # ==================================================
# # SMART RETRIEVAL
# # Strategy: ensemble | k=4 | reorder=True
# # Question: What is the procedure for filing a complaint against the ban...
# # ==================================================
# # Detected relevant document: complaint_handling_policy.pdf (score: 1)

# # Reordered 4 chunks — best at start and end

# # Final context: 2051 chars from 4 chunks
# # Sources: ['complaint_handling_policy.pdf', 'complaint_handling_policy.pdf', 'complaint_handling_policy.pdf', 'complaint_handling_policy.pdf']

# # Answer:
# # **Procedure for filing a complaint against ABC Bank**

# # 1. **Choose a filing channel**  
# #    - **Phone**: Call 0800‑ABCBANK (available 24 h/7 d).  
# #    - **Branch**: Visit any ABC Bank branch and speak to a Customer Service Officer.  
# #    - **Email**: Send an email to complaints@abcbank.com (response within 24 h).  
# #    - **Mobile App**: Open the ABC Bank app → Help section → File Complaint.  
# #    - **Website**: Go to abcbank.com/complaints and fill out the online form.  
# #    - **Written**: Mail a letter to the Complaints Department, Head Office, Karachi.

# # 2. **Provide the required information**  
# #    - Full name as per CNIC  
# #    - CNIC number  
# #    - Account number or card number  
# #    - Contact number and email address  
# #    - Clear description of the complaint  
# #    - Date and location of the incident  
# #    - Any reference numbers related to the issue  
# #    - Supporting documents (if available)

# # 3. **After registration**  
# #    - A complaint reference number is issued immediately.  
# #    - You will receive an SMS confirmation within 1 hour and an email confirmation within 2 hours.  
# #    - Use this reference number for all follow‑up inquiries.

# # Follow these steps to file a complaint and ensure it is processed efficiently.

# # ============================================================
# # DEBUGGING: How much does a RAAST transfer cost?
# # ============================================================

# # --- STEP 1: RETRIEVAL ---

# # ==================================================
# # SMART RETRIEVAL
# # Strategy: ensemble | k=4 | reorder=True
# # Question: How much does a RAAST transfer cost?...
# # ==================================================
# # Detected relevant document: schedule_of_charges.pdf (score: 3)

# # Reordered 4 chunks — best at start and end

# # Final context: 1220 chars from 4 chunks
# # Sources: ['schedule_of_charges.pdf', 'schedule_of_charges.pdf', 'schedule_of_charges.pdf', 'schedule_of_charges.pdf']

# # Retrieved 4 chunks:

# #   Chunk 1:
# #   Source: schedule_of_charges.pdf
# #   Page: 1
# #   Preview: Mini statement at own ATM: Free up to 2 per month 
# # Mini statement above 2: Rs. 25 per transaction 
 

# #   Chunk 2:
# #   Source: schedule_of_charges.pdf
# #   Page: 2
# #   Preview: Profit and loss certificate: Rs. 500 
# # Reference letter: Rs. 1,000 
# # Swift message charges outward: Rs

# #   Chunk 3:
# #   Source: schedule_of_charges.pdf
# #   Page: 2
# #   Preview: Balance inquiry at other bank ATM: Rs. 18 per transaction 
# # Maximum 5 free transactions per month for

# #   Chunk 4:
# #   Source: schedule_of_charges.pdf
# #   Page: 1
# #   Preview: IBFT above Rs. 500,000: Rs. 500 per transaction 
# # Internal fund transfer within ABC Bank: Free unlimi

# # --- STEP 2: AUGMENTATION ---
# # Context length: 1220 characters
# # Context preview:
# # [Document: schedule_of_charges.pdf | Page: 1]
# # Mini statement at own ATM: Free up to 2 per month 
# # Mini statement above 2: Rs. 25 per transaction 
 
# # 3.2 Other Bank ATM in Pakistan 
# # Cash withdrawal at other bank ATM: Rs. 25 per transaction

# # ---

# # [Document: schedule_of_charges.pdf | Page: 2]
# # Profit and loss certificate: Rs. 500 
# # Reference letter: Rs. 1,000 
# # Swift message charges outward: Rs. 1,500 
 
# # ...

# # --- STEP 3: GENERATION ---

# # ==================================================
# # SMART RETRIEVAL
# # Strategy: ensemble | k=4 | reorder=True
# # Question: How much does a RAAST transfer cost?...
# # ==================================================
# # Detected relevant document: schedule_of_charges.pdf (score: 3)

# # Reordered 4 chunks — best at start and end

# # Final context: 1220 chars from 4 chunks
# # Sources: ['schedule_of_charges.pdf', 'schedule_of_charges.pdf', 'schedule_of_charges.pdf', 'schedule_of_charges.pdf']

# # Answer:
# # RAAST transfer: Free unlimited.
# (venv) PS D:\Agentic AI\AI-powered bank customer-support assistant\AI-powered-bank-customer-support-assistant> 

# (venv) PS D:\Agentic AI\AI-powered bank customer-support assistant\AI-powered-bank-customer-support-assistant> python main.py
# warning: The `fitz` API is deprecated and will be removed in future. Use `import pymupdf` instead.
# USER_AGENT environment variable not set, consider setting it to identify your requests.

# ============================================================
# STEP 1 — LOADING DOCUMENTS
# ============================================================
# Normal PDF detected: account_opening_policy.pdf
# Loaded 2 pages from account_opening_policy.pdf
# Normal PDF detected: complaint_handling_policy.pdf
# Loaded 3 pages from complaint_handling_policy.pdf
# Normal PDF detected: credit_card_policy.pdf
# Loaded 3 pages from credit_card_policy.pdf
# Normal PDF detected: customer_faq.pdf
# Loaded 3 pages from customer_faq.pdf
# Normal PDF detected: loan_policy.pdf
# Loaded 3 pages from loan_policy.pdf
# Normal PDF detected: schedule_of_charges.pdf
# Loaded 2 pages from schedule_of_charges.pdf
# Scanned PDF detected: spytm_Scanned.pdf
# Running OCR on: spytm_Scanned.pdf
#   OCR page 1 of 1

# Total documents loaded: 17
# Total documents loaded: 17

# ============================================================
# STEP 2 — CLEANING DOCUMENTS
# ============================================================
# Cleaned 17 documents
# Removed 0 documents that were too short after cleaning
# Total documents after cleaning: 17

# ============================================================
# STEP 4 — INCREMENTAL INDEXING
# ============================================================

# Incremental indexing summary:
#   New documents:     0
#   Updated documents: 0
#   Skipped (unchanged): 7
#   Skipped: ['account_opening_policy.pdf', 'complaint_handling_policy.pdf', 'credit_card_policy.pdf', 'customer_faq.pdf', 'loan_policy.pdf', 'schedule_of_charges.pdf', 'spytm_Scanned.pdf']

# All documents already indexed — loading existing store
# D:\Agentic AI\AI-powered bank customer-support assistant\AI-powered-bank-customer-support-assistant\retrieval\embedder.py:15: LangChainDeprecationWarning: The class `HuggingFaceEmbeddings` was deprecated in LangChain 0.2.2 and will be removed in 1.0. An updated version of the class exists in the `langchain-huggingface package and should be used instead. To use it run `pip install -U `langchain-huggingface` and import as `from `langchain_huggingface import HuggingFaceEmbeddings``.
#   return HuggingFaceEmbeddings(
# Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
# Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 103/103 [00:00<00:00, 4146.28it/s]
# D:\Agentic AI\AI-powered bank customer-support assistant\AI-powered-bank-customer-support-assistant\retrieval\store.py:17: LangChainDeprecationWarning: The class `Chroma` was deprecated in LangChain 0.2.9 and will be removed in 1.0. An updated version of the class exists in the `langchain-chroma package and should be used instead. To use it run `pip install -U `langchain-chroma` and import as `from `langchain_chroma import Chroma``.
#   return Chroma(
# Total chunks in vector store: 66
# Vector store ready
# account_opening_policy.pdf → policy
# complaint_handling_policy.pdf → policy
# credit_card_policy.pdf → policy
# customer_faq.pdf → faq
# loan_policy.pdf → policy
# schedule_of_charges.pdf → charges
# spytm_Scanned.pdf → general
# Recommended for policy: {'chunk_size': 600, 'overlap': 100}
# Recursive chunking: 2 docs → 7 chunks
# Recommended for policy: {'chunk_size': 600, 'overlap': 100}
# Recursive chunking: 3 docs → 11 chunks
# Recommended for policy: {'chunk_size': 600, 'overlap': 100}
# Recursive chunking: 3 docs → 8 chunks
# Recommended for faq: {'chunk_size': 400, 'overlap': 50}
# Recursive chunking: 3 docs → 18 chunks
# Recommended for policy: {'chunk_size': 600, 'overlap': 100}
# Recursive chunking: 3 docs → 8 chunks
# Recommended for charges: {'chunk_size': 300, 'overlap': 30}
# Recursive chunking: 2 docs → 13 chunks
# Recommended for general: {'chunk_size': 500, 'overlap': 50}
# Recursive chunking: 1 docs → 1 chunks

# Chunks prepared for BM25: 66

# ============================================================
# STEP 6 — SETTING UP LLM
# ============================================================
# LLM configured

# ============================================================
# STEP 7 — CREATING RAG PROMPT
# ============================================================
# RAG prompt created

# ============================================================
# STEP 9 — BUILDING RAG CHAIN
# ============================================================
# RAG chain ready

# ============================================================
# DEBUGGING: What is the minimum balance required for a savings account?
# ============================================================

# --- STEP 1: RETRIEVAL ---

# ==================================================
# SMART RETRIEVAL
# Strategy: ensemble | k=4 | reorder=True
# Question: What is the minimum balance required for a savings account?...
# ==================================================
# Detected relevant document: account_opening_policy.pdf (score: 2)

# Reordered 4 chunks — best at start and end

# Final context: 2493 chars from 4 chunks
# Sources: ['account_opening_policy.pdf', 'account_opening_policy.pdf', 'account_opening_policy.pdf', 'account_opening_policy.pdf']

# Retrieved 4 chunks:

#   Chunk 1:
#   Source: account_opening_policy.pdf
#   Page: 1
#   Preview: Foreign Currency Account: USD 100 minimum balance 
 
# Below minimum balance charge: Rs. 500 per month

#   Chunk 2:
#   Source: account_opening_policy.pdf
#   Page: 2
#   Preview: Debit card issued within 7 working days 
# Online banking activated within 24 hours 
# Cheque book issue

#   Chunk 3:
#   Source: account_opening_policy.pdf
#   Page: 1
#   Preview: ABC BANK — ACCOUNT OPENING POLICY 
# Version 1.0 | Effective Date: January 2024 
 
# SECTION 1 — TYPES O

#   Chunk 4:
#   Source: account_opening_policy.pdf
#   Page: 1
#   Preview: 1.2 Business Accounts 
# Current Account for Business — unlimited transactions, no profit 
# Business Sa

# --- STEP 2: AUGMENTATION ---
# Context length: 2493 characters
# Context preview:
# [Document: account_opening_policy.pdf | Page: 1]
# Foreign Currency Account: USD 100 minimum balance 
 
# Below minimum balance charge: Rs. 500 per month for savings 
# Below minimum balance charge: Rs. 1,000 per month for current 
 
# SECTION 3 — PROFIT RATES 
 
# Regular Savings Account: 5% per annum 
# Premium Savings Account: 7% per annum 
# Basic Banking Account: 0% profit 
# Profit credited: monthly on last...

# --- STEP 3: GENERATION ---

# ==================================================
# SMART RETRIEVAL
# Strategy: ensemble | k=4 | reorder=True
# Question: What is the minimum balance required for a savings account?...
# ==================================================
# Detected relevant document: account_opening_policy.pdf (score: 2)

# Reordered 4 chunks — best at start and end

# Final context: 2493 chars from 4 chunks
# Sources: ['account_opening_policy.pdf', 'account_opening_policy.pdf', 'account_opening_policy.pdf', 'account_opening_policy.pdf']

# Answer:
# The minimum balance required for a Regular Savings Account is **Rs. 10,000**.

# ============================================================
# DEBUGGING: Which documents do I need to open a bank account?
# ============================================================

# --- STEP 1: RETRIEVAL ---

# ==================================================
# SMART RETRIEVAL
# Strategy: ensemble | k=4 | reorder=True
# Question: Which documents do I need to open a bank account?...
# ==================================================
# Detected relevant document: account_opening_policy.pdf (score: 2)

# Reordered 4 chunks — best at start and end

# Final context: 2132 chars from 4 chunks
# Sources: ['account_opening_policy.pdf', 'account_opening_policy.pdf', 'account_opening_policy.pdf', 'account_opening_policy.pdf']

# Retrieved 4 chunks:

#   Chunk 1:
#   Source: account_opening_policy.pdf
#   Page: 1
#   Preview: 1.2 Business Accounts 
# Current Account for Business — unlimited transactions, no profit 
# Business Sa

#   Chunk 2:
#   Source: account_opening_policy.pdf
#   Page: 2
#   Preview: Minor original B-Form 
# Parent or guardian original CNIC 
# Proof of relationship — birth certificate 


#   Chunk 3:
#   Source: account_opening_policy.pdf
#   Page: 1
#   Preview: Original CNIC — mandatory 
# Utility bill not older than 3 months — proof of address 
# Passport size ph

#   Chunk 4:
#   Source: account_opening_policy.pdf
#   Page: 1
#   Preview: ABC BANK — ACCOUNT OPENING POLICY 
# Version 1.0 | Effective Date: January 2024 
 
# SECTION 1 — TYPES O

# --- STEP 2: AUGMENTATION ---
# Context length: 2132 characters
# Context preview:
# [Document: account_opening_policy.pdf | Page: 1]
# 1.2 Business Accounts 
# Current Account for Business — unlimited transactions, no profit 
# Business Savings Account — limited transactions with profit 
# Corporate Account — for registered companies 
 
# SECTION 2 — MINIMUM BALANCE REQUIREMENTS 
 
# Basic Banking Account: Rs. 0 minimum balance 
# Regular Savings Account: Rs. 10,000 minimum balance 
# Premium Sa...

# --- STEP 3: GENERATION ---

# ==================================================
# SMART RETRIEVAL
# Strategy: ensemble | k=4 | reorder=True
# Question: Which documents do I need to open a bank account?...
# ==================================================
# Detected relevant document: account_opening_policy.pdf (score: 2)

# Reordered 4 chunks — best at start and end

# Final context: 2132 chars from 4 chunks
# Sources: ['account_opening_policy.pdf', 'account_opening_policy.pdf', 'account_opening_policy.pdf', 'account_opening_policy.pdf']

# Answer:
# To open an account at ABC Bank you’ll need to provide the following documents:

# | Account type | Required documents |
# |--------------|--------------------|
# | **All accounts (individual, savings, current, foreign‑currency, etc.)** | • Original CNIC (mandatory)<br>• Utility bill (not older than 3 months) – proof of address<br>• 2 passport‑size photographs<br>• Source of income declaration form (mandatory)<br>• Next‑of‑kin details (mandatory) |
# | **Minor accounts** | • Minor’s original B‑Form<br>• Parent or guardian’s original CNIC<br>• Proof of relationship (birth certificate)<br>• Guardian’s utility bill |
# | **Business accounts** | • Business owner’s CNIC<br>• Business registration certificate<br>• NTN certificate<br>• Board resolution for authorized signatories<br>• Partnership deed (if applicable)<br>• Memorandum and articles of association (for companies) |

# Make sure all documents are original and up‑to‑date. Once you submit them, the account will be activated within 24 hours and a debit card will be issued within 7 working days.

# ============================================================
# DEBUGGING: How much is the IBFT transfer charge?
# ============================================================

# --- STEP 1: RETRIEVAL ---

# ==================================================
# SMART RETRIEVAL
# Strategy: ensemble | k=4 | reorder=True
# Question: How much is the IBFT transfer charge?...
# ==================================================
# Detected relevant document: schedule_of_charges.pdf (score: 3)

# Reordered 4 chunks — best at start and end

# Final context: 1314 chars from 4 chunks
# Sources: ['schedule_of_charges.pdf', 'schedule_of_charges.pdf', 'schedule_of_charges.pdf', 'schedule_of_charges.pdf']

# Retrieved 4 chunks:

#   Chunk 1:
#   Source: schedule_of_charges.pdf
#   Page: 1
#   Preview: Cheque return charges inward: Rs. 1,000 per cheque 
# Cheque return charges outward: Rs. 500 per chequ

#   Chunk 2:
#   Source: schedule_of_charges.pdf
#   Page: 2
#   Preview: Foreign currency conversion rate: SBP rate plus 3.5% 
 
# SECTION 4 — CARD CHARGES 
 
# Debit card issua

#   Chunk 3:
#   Source: schedule_of_charges.pdf
#   Page: 2
#   Preview: Profit and loss certificate: Rs. 500 
# Reference letter: Rs. 1,000 
# Swift message charges outward: Rs

#   Chunk 4:
#   Source: schedule_of_charges.pdf
#   Page: 1
#   Preview: IBFT above Rs. 500,000: Rs. 500 per transaction 
# Internal fund transfer within ABC Bank: Free unlimi

# --- STEP 2: AUGMENTATION ---
# Context length: 1314 characters
# Context preview:
# [Document: schedule_of_charges.pdf | Page: 1]
# Cheque return charges inward: Rs. 1,000 per cheque 
# Cheque return charges outward: Rs. 500 per cheque 
# Stop payment instruction: Rs. 500 
 
# 2.3 Online and Digital Transactions 
# IBFT up to Rs. 25,000: Rs. 200 per transaction 
# IBFT Rs. 25,001 to Rs. 500,000: Rs. 300 per transaction

# ---

# [Document: schedule_of_charges.pdf | Page: 2]
# Foreign currency conv...

# --- STEP 3: GENERATION ---

# ==================================================
# SMART RETRIEVAL
# Strategy: ensemble | k=4 | reorder=True
# Question: How much is the IBFT transfer charge?...
# ==================================================
# Detected relevant document: schedule_of_charges.pdf (score: 3)

# Reordered 4 chunks — best at start and end

# Final context: 1314 chars from 4 chunks
# Sources: ['schedule_of_charges.pdf', 'schedule_of_charges.pdf', 'schedule_of_charges.pdf', 'schedule_of_charges.pdf']

# Answer:
# IBFT transfer charges vary by the amount transferred:

# | Transfer amount | Charge per transaction |
# |-----------------|------------------------|
# | Up to Rs. 25,000 | Rs. 200 |
# | Rs. 25,001 – Rs. 500,000 | Rs. 300 |
# | Above Rs. 500,000 | Rs. 500 |

# ============================================================
# DEBUGGING: What is the procedure for filing a complaint against the bank?
# ============================================================

# --- STEP 1: RETRIEVAL ---

# ==================================================
# SMART RETRIEVAL
# Strategy: ensemble | k=4 | reorder=True
# Question: What is the procedure for filing a complaint against the ban...
# ==================================================
# Detected relevant document: complaint_handling_policy.pdf (score: 1)

# Reordered 4 chunks — best at start and end

# Final context: 2051 chars from 4 chunks
# Sources: ['complaint_handling_policy.pdf', 'complaint_handling_policy.pdf', 'complaint_handling_policy.pdf', 'complaint_handling_policy.pdf']

# Retrieved 4 chunks:

#   Chunk 1:
#   Source: complaint_handling_policy.pdf
#   Page: 3
#   Preview: Right to escalate if not satisfied with resolution 
# Right to approach Banking Mohtasib if internal r

#   Chunk 2:
#   Source: complaint_handling_policy.pdf
#   Page: 1
#   Preview: CNIC number 
# Account number or card number 
# Contact number and email address 
# Clear description of t

#   Chunk 3:
#   Source: complaint_handling_policy.pdf
#   Page: 1
#   Preview: ABC BANK — COMPLAINT HANDLING POLICY 
# Version 1.0 | Effective Date: January 2024 
 
# SECTION 1 — POLI

#   Chunk 4:
#   Source: complaint_handling_policy.pdf
#   Page: 1
#   Preview: 2.1 Available Channels 
# Phone: Call 0800-ABCBANK available 24 hours 7 days a week 
# Branch: Visit any

# --- STEP 2: AUGMENTATION ---
# Context length: 2051 characters
# Context preview:
# [Document: complaint_handling_policy.pdf | Page: 3]
# Right to escalate if not satisfied with resolution 
# Right to approach Banking Mohtasib if internal resolution fails 
# Right to confidentiality of complaint details

# ---

# [Document: complaint_handling_policy.pdf | Page: 1]
# CNIC number 
# Account number or card number 
# Contact number and email address 
# Clear description of the complaint 
# Date and loca...

# --- STEP 3: GENERATION ---

# ==================================================
# SMART RETRIEVAL
# Strategy: ensemble | k=4 | reorder=True
# Question: What is the procedure for filing a complaint against the ban...
# ==================================================
# Detected relevant document: complaint_handling_policy.pdf (score: 1)

# Reordered 4 chunks — best at start and end

# Final context: 2051 chars from 4 chunks
# Sources: ['complaint_handling_policy.pdf', 'complaint_handling_policy.pdf', 'complaint_handling_policy.pdf', 'complaint_handling_policy.pdf']

# Answer:
# **Procedure for filing a complaint against ABC Bank**

# 1. **Choose a filing channel**  
#    - **Phone**: Call 0800‑ABCBANK (available 24 h/7 d).  
#    - **Branch**: Visit any ABC Bank branch and speak to a Customer Service Officer.  
#    - **Email**: Send an email to complaints@abcbank.com (response within 24 h).  
#    - **Mobile App**: Open the ABC Bank app → Help section → File Complaint.  
#    - **Website**: Go to abcbank.com/complaints and fill out the online form.  
#    - **Written**: Mail a letter to the Complaints Department, Head Office, Karachi.

# 2. **Provide the required information**  
#    - Full name as per CNIC  
#    - CNIC number  
#    - Account number or card number  
#    - Contact number and email address  
#    - Clear description of the complaint  
#    - Date and location of the incident  
#    - Any reference numbers related to the issue  
#    - Supporting documents (if available)

# 3. **After registration**  
#    - A complaint reference number is issued immediately.  
#    - You will receive an SMS confirmation within 1 hour and an email confirmation within 2 hours.  
#    - Use this reference number for all follow‑up inquiries.

# Follow these steps to file a complaint and ensure it is processed efficiently.

# ============================================================
# DEBUGGING: How much does a RAAST transfer cost?
# ============================================================

# --- STEP 1: RETRIEVAL ---

# ==================================================
# SMART RETRIEVAL
# Strategy: ensemble | k=4 | reorder=True
# Question: How much does a RAAST transfer cost?...
# ==================================================
# Detected relevant document: schedule_of_charges.pdf (score: 3)

# Reordered 4 chunks — best at start and end

# Final context: 1220 chars from 4 chunks
# Sources: ['schedule_of_charges.pdf', 'schedule_of_charges.pdf', 'schedule_of_charges.pdf', 'schedule_of_charges.pdf']

# Retrieved 4 chunks:

#   Chunk 1:
#   Source: schedule_of_charges.pdf
#   Page: 1
#   Preview: Mini statement at own ATM: Free up to 2 per month 
# Mini statement above 2: Rs. 25 per transaction 
 

#   Chunk 2:
#   Source: schedule_of_charges.pdf
#   Page: 2
#   Preview: Profit and loss certificate: Rs. 500 
# Reference letter: Rs. 1,000 
# Swift message charges outward: Rs

#   Chunk 3:
#   Source: schedule_of_charges.pdf
#   Page: 2
#   Preview: Balance inquiry at other bank ATM: Rs. 18 per transaction 
# Maximum 5 free transactions per month for

#   Chunk 4:
#   Source: schedule_of_charges.pdf
#   Page: 1
#   Preview: IBFT above Rs. 500,000: Rs. 500 per transaction 
# Internal fund transfer within ABC Bank: Free unlimi

# --- STEP 2: AUGMENTATION ---
# Context length: 1220 characters
# Context preview:
# [Document: schedule_of_charges.pdf | Page: 1]
# Mini statement at own ATM: Free up to 2 per month 
# Mini statement above 2: Rs. 25 per transaction 
 
# 3.2 Other Bank ATM in Pakistan 
# Cash withdrawal at other bank ATM: Rs. 25 per transaction

# ---

# [Document: schedule_of_charges.pdf | Page: 2]
# Profit and loss certificate: Rs. 500 
# Reference letter: Rs. 1,000 
# Swift message charges outward: Rs. 1,500 
 
# ...

# --- STEP 3: GENERATION ---

# ==================================================
# SMART RETRIEVAL
# Strategy: ensemble | k=4 | reorder=True
# Question: How much does a RAAST transfer cost?...
# ==================================================
# Detected relevant document: schedule_of_charges.pdf (score: 3)

# Reordered 4 chunks — best at start and end

# Final context: 1220 chars from 4 chunks
# Sources: ['schedule_of_charges.pdf', 'schedule_of_charges.pdf', 'schedule_of_charges.pdf', 'schedule_of_charges.pdf']

# Answer:
# RAAST transfer: Free unlimited.
# (venv) PS D:\Agentic AI\AI-powered bank customer-support assistant\AI-powered-bank-customer-support-assistant> 

# (venv) PS D:\Agentic AI\AI-powered bank customer-support assistant\AI-powered-bank-customer-support-assistant> python main.py
# warning: The `fitz` API is deprecated and will be removed in future. Use `import pymupdf` instead.
# USER_AGENT environment variable not set, consider setting it to identify your requests.

# ============================================================
# STEP 1 — LOADING DOCUMENTS
# ============================================================
# Normal PDF detected: account_opening_policy.pdf
# Loaded 2 pages from account_opening_policy.pdf
# Normal PDF detected: complaint_handling_policy.pdf
# Loaded 3 pages from complaint_handling_policy.pdf
# Normal PDF detected: credit_card_policy.pdf
# Loaded 3 pages from credit_card_policy.pdf
# Normal PDF detected: customer_faq.pdf
# Loaded 3 pages from customer_faq.pdf
# Normal PDF detected: loan_policy.pdf
# Loaded 3 pages from loan_policy.pdf
# Normal PDF detected: schedule_of_charges.pdf
# Loaded 2 pages from schedule_of_charges.pdf
# Scanned PDF detected: spytm_Scanned.pdf
# Running OCR on: spytm_Scanned.pdf
#   OCR page 1 of 1

# Total documents loaded: 17
# Total documents loaded: 17

# ============================================================
# STEP 2 — CLEANING DOCUMENTS
# ============================================================
# Cleaned 17 documents
# Removed 0 documents that were too short after cleaning
# Total documents after cleaning: 17

# ============================================================
# STEP 4 — INCREMENTAL INDEXING
# ============================================================

# Incremental indexing summary:
#   New documents:     0
#   Updated documents: 0
#   Skipped (unchanged): 7
#   Skipped: ['account_opening_policy.pdf', 'complaint_handling_policy.pdf', 'credit_card_policy.pdf', 'customer_faq.pdf', 'loan_policy.pdf', 'schedule_of_charges.pdf', 'spytm_Scanned.pdf']

# All documents already indexed — loading existing store
# D:\Agentic AI\AI-powered bank customer-support assistant\AI-powered-bank-customer-support-assistant\retrieval\embedder.py:15: LangChainDeprecationWarning: The class `HuggingFaceEmbeddings` was deprecated in LangChain 0.2.2 and will be removed in 1.0. An updated version of the class exists in the `langchain-huggingface package and should be used instead. To use it run `pip install -U `langchain-huggingface` and import as `from `langchain_huggingface import HuggingFaceEmbeddings``.
#   return HuggingFaceEmbeddings(
# Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
# Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 103/103 [00:00<00:00, 4973.67it/s]
# D:\Agentic AI\AI-powered bank customer-support assistant\AI-powered-bank-customer-support-assistant\retrieval\store.py:17: LangChainDeprecationWarning: The class `Chroma` was deprecated in LangChain 0.2.9 and will be removed in 1.0. An updated version of the class exists in the `langchain-chroma package and should be used instead. To use it run `pip install -U `langchain-chroma` and import as `from `langchain_chroma import Chroma``.
#   return Chroma(
# Total chunks in vector store: 66
# Vector store ready
# account_opening_policy.pdf → policy
# complaint_handling_policy.pdf → policy
# credit_card_policy.pdf → policy
# customer_faq.pdf → faq
# loan_policy.pdf → policy
# schedule_of_charges.pdf → charges
# spytm_Scanned.pdf → general
# Recommended for policy: {'chunk_size': 600, 'overlap': 100}
# Recursive chunking: 2 docs → 7 chunks
# Recommended for policy: {'chunk_size': 600, 'overlap': 100}
# Recursive chunking: 3 docs → 11 chunks
# Recommended for policy: {'chunk_size': 600, 'overlap': 100}
# Recursive chunking: 3 docs → 8 chunks
# Recommended for faq: {'chunk_size': 400, 'overlap': 50}
# Recursive chunking: 3 docs → 18 chunks
# Recommended for policy: {'chunk_size': 600, 'overlap': 100}
# Recursive chunking: 3 docs → 8 chunks
# Recommended for charges: {'chunk_size': 300, 'overlap': 30}
# Recursive chunking: 2 docs → 13 chunks
# Recommended for general: {'chunk_size': 500, 'overlap': 50}
# Recursive chunking: 1 docs → 1 chunks

# Chunks prepared for BM25: 66

# ============================================================
# STEP 6 — SETTING UP LLM
# ============================================================
# LLM configured

# ============================================================
# STEP 7 — CREATING RAG PROMPT
# ============================================================
# RAG prompt created

# ============================================================
# STEP 9 — BUILDING RAG CHAIN
# ============================================================
# RAG chain ready

# ============================================================
# DEBUGGING: What is the minimum balance required for a savings account?
# ============================================================

# --- STEP 1: RETRIEVAL ---

# ==================================================
# SMART RETRIEVAL
# Strategy: ensemble | k=4 | reorder=True
# Question: What is the minimum balance required for a savings account?...
# ==================================================
# Detected relevant document: account_opening_policy.pdf (score: 2)

# Reordered 4 chunks — best at start and end

# Final context: 2493 chars from 4 chunks
# Sources: ['account_opening_policy.pdf', 'account_opening_policy.pdf', 'account_opening_policy.pdf', 'account_opening_policy.pdf']

# Retrieved 4 chunks:

#   Chunk 1:
#   Source: account_opening_policy.pdf
#   Page: 1
#   Preview: Foreign Currency Account: USD 100 minimum balance 
 
# Below minimum balance charge: Rs. 500 per month

#   Chunk 2:
#   Source: account_opening_policy.pdf
#   Page: 2
#   Preview: Debit card issued within 7 working days 
# Online banking activated within 24 hours 
# Cheque book issue

#   Chunk 3:
#   Source: account_opening_policy.pdf
#   Page: 1
#   Preview: ABC BANK — ACCOUNT OPENING POLICY 
# Version 1.0 | Effective Date: January 2024 
 
# SECTION 1 — TYPES O

#   Chunk 4:
#   Source: account_opening_policy.pdf
#   Page: 1
#   Preview: 1.2 Business Accounts 
# Current Account for Business — unlimited transactions, no profit 
# Business Sa

# --- STEP 2: AUGMENTATION ---
# Context length: 2493 characters
# Context preview:
# [Document: account_opening_policy.pdf | Page: 1]
# Foreign Currency Account: USD 100 minimum balance 
 
# Below minimum balance charge: Rs. 500 per month for savings 
# Below minimum balance charge: Rs. 1,000 per month for current 
 
# SECTION 3 — PROFIT RATES 
 
# Regular Savings Account: 5% per annum 
# Premium Savings Account: 7% per annum 
# Basic Banking Account: 0% profit 
# Profit credited: monthly on last...

# --- STEP 3: GENERATION ---

# ==================================================
# SMART RETRIEVAL
# Strategy: ensemble | k=4 | reorder=True
# Question: What is the minimum balance required for a savings account?...
# ==================================================
# Detected relevant document: account_opening_policy.pdf (score: 2)

# Reordered 4 chunks — best at start and end

# Final context: 2493 chars from 4 chunks
# Sources: ['account_opening_policy.pdf', 'account_opening_policy.pdf', 'account_opening_policy.pdf', 'account_opening_policy.pdf']

# Answer:
# The minimum balance required for a Regular Savings Account is **Rs. 10,000**.

# ============================================================
# DEBUGGING: Which documents do I need to open a bank account?
# ============================================================

# --- STEP 1: RETRIEVAL ---

# ==================================================
# SMART RETRIEVAL
# Strategy: ensemble | k=4 | reorder=True
# Question: Which documents do I need to open a bank account?...
# ==================================================
# Detected relevant document: account_opening_policy.pdf (score: 2)

# Reordered 4 chunks — best at start and end

# Final context: 2132 chars from 4 chunks
# Sources: ['account_opening_policy.pdf', 'account_opening_policy.pdf', 'account_opening_policy.pdf', 'account_opening_policy.pdf']

# Retrieved 4 chunks:

#   Chunk 1:
#   Source: account_opening_policy.pdf
#   Page: 1
#   Preview: 1.2 Business Accounts 
# Current Account for Business — unlimited transactions, no profit 
# Business Sa

#   Chunk 2:
#   Source: account_opening_policy.pdf
#   Page: 2
#   Preview: Minor original B-Form 
# Parent or guardian original CNIC 
# Proof of relationship — birth certificate 


#   Chunk 3:
#   Source: account_opening_policy.pdf
#   Page: 1
#   Preview: Original CNIC — mandatory 
# Utility bill not older than 3 months — proof of address 
# Passport size ph

#   Chunk 4:
#   Source: account_opening_policy.pdf
#   Page: 1
#   Preview: ABC BANK — ACCOUNT OPENING POLICY 
# Version 1.0 | Effective Date: January 2024 
 
# SECTION 1 — TYPES O

# --- STEP 2: AUGMENTATION ---
# Context length: 2132 characters
# Context preview:
# [Document: account_opening_policy.pdf | Page: 1]
# 1.2 Business Accounts 
# Current Account for Business — unlimited transactions, no profit 
# Business Savings Account — limited transactions with profit 
# Corporate Account — for registered companies 
 
# SECTION 2 — MINIMUM BALANCE REQUIREMENTS 
 
# Basic Banking Account: Rs. 0 minimum balance 
# Regular Savings Account: Rs. 10,000 minimum balance 
# Premium Sa...

# --- STEP 3: GENERATION ---

# ==================================================
# SMART RETRIEVAL
# Strategy: ensemble | k=4 | reorder=True
# Question: Which documents do I need to open a bank account?...
# ==================================================
# Detected relevant document: account_opening_policy.pdf (score: 2)

# Reordered 4 chunks — best at start and end

# Final context: 2132 chars from 4 chunks
# Sources: ['account_opening_policy.pdf', 'account_opening_policy.pdf', 'account_opening_policy.pdf', 'account_opening_policy.pdf']

# Answer:
# To open an account at ABC Bank you’ll need to provide the following documents:

# | Account type | Required documents |
# |--------------|--------------------|
# | **All accounts (individual, savings, current, foreign‑currency, etc.)** | • Original CNIC (mandatory)<br>• Utility bill (not older than 3 months) – proof of address<br>• 2 passport‑size photographs<br>• Source of income declaration form (mandatory)<br>• Next‑of‑kin details (mandatory) |
# | **Minor accounts** | • Minor’s original B‑Form<br>• Parent or guardian’s original CNIC<br>• Proof of relationship (birth certificate)<br>• Guardian’s utility bill |
# | **Business accounts** | • Business owner’s CNIC<br>• Business registration certificate<br>• NTN certificate<br>• Board resolution for authorized signatories<br>• Partnership deed (if applicable)<br>• Memorandum and articles of association (for companies) |

# Make sure all documents are original and up‑to‑date. Once you submit them, the account will be activated within 24 hours.

# ============================================================
# DEBUGGING: How much is the IBFT transfer charge?
# ============================================================

# --- STEP 1: RETRIEVAL ---

# ==================================================
# SMART RETRIEVAL
# Strategy: ensemble | k=4 | reorder=True
# Question: How much is the IBFT transfer charge?...
# ==================================================
# Detected relevant document: schedule_of_charges.pdf (score: 3)

# Reordered 4 chunks — best at start and end

# Final context: 1314 chars from 4 chunks
# Sources: ['schedule_of_charges.pdf', 'schedule_of_charges.pdf', 'schedule_of_charges.pdf', 'schedule_of_charges.pdf']

# Retrieved 4 chunks:

#   Chunk 1:
#   Source: schedule_of_charges.pdf
#   Page: 1
#   Preview: Cheque return charges inward: Rs. 1,000 per cheque 
# Cheque return charges outward: Rs. 500 per chequ

#   Chunk 2:
#   Source: schedule_of_charges.pdf
#   Page: 2
#   Preview: Foreign currency conversion rate: SBP rate plus 3.5% 
 
# SECTION 4 — CARD CHARGES 
 
# Debit card issua

#   Chunk 3:
#   Source: schedule_of_charges.pdf
#   Page: 2
#   Preview: Profit and loss certificate: Rs. 500 
# Reference letter: Rs. 1,000 
# Swift message charges outward: Rs

#   Chunk 4:
#   Source: schedule_of_charges.pdf
#   Page: 1
#   Preview: IBFT above Rs. 500,000: Rs. 500 per transaction 
# Internal fund transfer within ABC Bank: Free unlimi

# --- STEP 2: AUGMENTATION ---
# Context length: 1314 characters
# Context preview:
# [Document: schedule_of_charges.pdf | Page: 1]
# Cheque return charges inward: Rs. 1,000 per cheque 
# Cheque return charges outward: Rs. 500 per cheque 
# Stop payment instruction: Rs. 500 
 
# 2.3 Online and Digital Transactions 
# IBFT up to Rs. 25,000: Rs. 200 per transaction 
# IBFT Rs. 25,001 to Rs. 500,000: Rs. 300 per transaction

# ---

# [Document: schedule_of_charges.pdf | Page: 2]
# Foreign currency conv...

# --- STEP 3: GENERATION ---

# ==================================================
# SMART RETRIEVAL
# Strategy: ensemble | k=4 | reorder=True
# Question: How much is the IBFT transfer charge?...
# ==================================================
# Detected relevant document: schedule_of_charges.pdf (score: 3)

# Reordered 4 chunks — best at start and end

# Final context: 1314 chars from 4 chunks
# Sources: ['schedule_of_charges.pdf', 'schedule_of_charges.pdf', 'schedule_of_charges.pdf', 'schedule_of_charges.pdf']

# Answer:
# IBFT transfer charges vary by the amount transferred:

# | Transfer amount | Charge per transaction |
# |-----------------|------------------------|
# | Up to Rs. 25,000 | Rs. 200 |
# | Rs. 25,001 – Rs. 500,000 | Rs. 300 |
# | Above Rs. 500,000 | Rs. 500 |

# ============================================================
# DEBUGGING: What is the procedure for filing a complaint against the bank?
# ============================================================

# --- STEP 1: RETRIEVAL ---

# ==================================================
# SMART RETRIEVAL
# Strategy: ensemble | k=4 | reorder=True
# Question: What is the procedure for filing a complaint against the ban...
# ==================================================
# Detected relevant document: complaint_handling_policy.pdf (score: 1)

# Reordered 4 chunks — best at start and end

# Final context: 2051 chars from 4 chunks
# Sources: ['complaint_handling_policy.pdf', 'complaint_handling_policy.pdf', 'complaint_handling_policy.pdf', 'complaint_handling_policy.pdf']

# Retrieved 4 chunks:

#   Chunk 1:
#   Source: complaint_handling_policy.pdf
#   Page: 3
#   Preview: Right to escalate if not satisfied with resolution 
# Right to approach Banking Mohtasib if internal r

#   Chunk 2:
#   Source: complaint_handling_policy.pdf
#   Page: 1
#   Preview: CNIC number 
# Account number or card number 
# Contact number and email address 
# Clear description of t

#   Chunk 3:
#   Source: complaint_handling_policy.pdf
#   Page: 1
#   Preview: ABC BANK — COMPLAINT HANDLING POLICY 
# Version 1.0 | Effective Date: January 2024 
 
# SECTION 1 — POLI

#   Chunk 4:
#   Source: complaint_handling_policy.pdf
#   Page: 1
#   Preview: 2.1 Available Channels 
# Phone: Call 0800-ABCBANK available 24 hours 7 days a week 
# Branch: Visit any

# --- STEP 2: AUGMENTATION ---
# Context length: 2051 characters
# Context preview:
# [Document: complaint_handling_policy.pdf | Page: 3]
# Right to escalate if not satisfied with resolution 
# Right to approach Banking Mohtasib if internal resolution fails 
# Right to confidentiality of complaint details

# ---

# [Document: complaint_handling_policy.pdf | Page: 1]
# CNIC number 
# Account number or card number 
# Contact number and email address 
# Clear description of the complaint 
# Date and loca...

# --- STEP 3: GENERATION ---

# ==================================================
# SMART RETRIEVAL
# Strategy: ensemble | k=4 | reorder=True
# Question: What is the procedure for filing a complaint against the ban...
# ==================================================
# Detected relevant document: complaint_handling_policy.pdf (score: 1)

# Reordered 4 chunks — best at start and end

# Final context: 2051 chars from 4 chunks
# Sources: ['complaint_handling_policy.pdf', 'complaint_handling_policy.pdf', 'complaint_handling_policy.pdf', 'complaint_handling_policy.pdf']

# Answer:
# **Procedure for filing a complaint against ABC Bank**

# 1. **Choose a filing channel**  
#    - **Phone**: Call 0800‑ABCBANK (available 24 h/7 d).  
#    - **Branch**: Visit any ABC Bank branch and speak to a Customer Service Officer.  
#    - **Email**: Send an email to complaints@abcbank.com (response within 24 h).  
#    - **Mobile App**: Open the ABC Bank app → Help section → File Complaint.  
#    - **Website**: Go to abcbank.com/complaints and fill out the online form.  
#    - **Written**: Mail a letter to the Complaints Department, Head Office, Karachi.

# 2. **Provide the required information**  
#    - Full name as per CNIC  
#    - CNIC number  
#    - Account number or card number  
#    - Contact number and email address  
#    - Clear description of the complaint  
#    - Date and location of the incident  
#    - Any reference numbers related to the issue  
#    - Supporting documents (if available)

# 3. **After registration**  
#    - A complaint reference number is issued immediately.  
#    - You will receive an SMS confirmation within 1 hour and an email confirmation within 2 hours.  
#    - Use this reference number for all follow‑up inquiries.

# Follow these steps to file a complaint and ensure it is processed efficiently.

# ============================================================
# DEBUGGING: How much does a RAAST transfer cost?
# ============================================================

# --- STEP 1: RETRIEVAL ---

# ==================================================
# SMART RETRIEVAL
# Strategy: ensemble | k=4 | reorder=True
# Question: How much does a RAAST transfer cost?...
# ==================================================
# Detected relevant document: schedule_of_charges.pdf (score: 3)

# Reordered 4 chunks — best at start and end

# Final context: 1220 chars from 4 chunks
# Sources: ['schedule_of_charges.pdf', 'schedule_of_charges.pdf', 'schedule_of_charges.pdf', 'schedule_of_charges.pdf']

# Retrieved 4 chunks:

#   Chunk 1:
#   Source: schedule_of_charges.pdf
#   Page: 1
#   Preview: Mini statement at own ATM: Free up to 2 per month 
# Mini statement above 2: Rs. 25 per transaction 
 

#   Chunk 2:
#   Source: schedule_of_charges.pdf
#   Page: 2
#   Preview: Profit and loss certificate: Rs. 500 
# Reference letter: Rs. 1,000 
# Swift message charges outward: Rs

#   Chunk 3:
#   Source: schedule_of_charges.pdf
#   Page: 2
#   Preview: Balance inquiry at other bank ATM: Rs. 18 per transaction 
# Maximum 5 free transactions per month for

#   Chunk 4:
#   Source: schedule_of_charges.pdf
#   Page: 1
#   Preview: IBFT above Rs. 500,000: Rs. 500 per transaction 
# Internal fund transfer within ABC Bank: Free unlimi

# --- STEP 2: AUGMENTATION ---
# Context length: 1220 characters
# Context preview:
# [Document: schedule_of_charges.pdf | Page: 1]
# Mini statement at own ATM: Free up to 2 per month 
# Mini statement above 2: Rs. 25 per transaction 
 
# 3.2 Other Bank ATM in Pakistan 
# Cash withdrawal at other bank ATM: Rs. 25 per transaction

# ---

# [Document: schedule_of_charges.pdf | Page: 2]
# Profit and loss certificate: Rs. 500 
# Reference letter: Rs. 1,000 
# Swift message charges outward: Rs. 1,500 
 
# ...

# --- STEP 3: GENERATION ---

# ==================================================
# SMART RETRIEVAL
# Strategy: ensemble | k=4 | reorder=True
# Question: How much does a RAAST transfer cost?...
# ==================================================
# Detected relevant document: schedule_of_charges.pdf (score: 3)

# Reordered 4 chunks — best at start and end

# Final context: 1220 chars from 4 chunks
# Sources: ['schedule_of_charges.pdf', 'schedule_of_charges.pdf', 'schedule_of_charges.pdf', 'schedule_of_charges.pdf']

# Answer:
# RAAST transfer: Free unlimited.
# (venv) PS D:\Agentic AI\AI-powered bank customer-support assistant\AI-powered-bank-customer-support-assistant> 
# (venv) PS D:\Agentic AI\AI-powered bank customer-support assistant\AI-powered-bank-customer-support-assistant> python main.py 
# warning: The `fitz` API is deprecated and will be removed in future. Use `import pymupdf` instead. 
# USER_AGENT environment variable not set, consider setting it to identify your requests. 
 
# ============================================================ 
# STEP 1 — LOADING DOCUMENTS 
# ============================================================ 
# Normal PDF detected: account_opening_policy.pdf 
# Loaded 2 pages from account_opening_policy.pdf 
# Normal PDF detected: complaint_handling_policy.pdf 
# Loaded 3 pages from complaint_handling_policy.pdf 
# Normal PDF detected: credit_card_policy.pdf 
# Loaded 3 pages from credit_card_policy.pdf 
# Normal PDF detected: customer_faq.pdf 
# Loaded 3 pages from customer_faq.pdf 
# Normal PDF detected: loan_policy.pdf 
# Loaded 3 pages from loan_policy.pdf 
# Normal PDF detected: schedule_of_charges.pdf 
# Loaded 2 pages from schedule_of_charges.pdf 
# Scanned PDF detected: spytm_Scanned.pdf 
# Running OCR on: spytm_Scanned.pdf 
#   OCR page 1 of 1 
 
# Total documents loaded: 17 
# Total documents loaded: 17 
 
# ============================================================ 
# STEP 2 — CLEANING DOCUMENTS 
# ============================================================ 
# Cleaned 17 documents 
# Removed 0 documents that were too short after cleaning 
# Total documents after cleaning: 17 
 
# ============================================================ 
# STEP 4 — INCREMENTAL INDEXING 
# ============================================================ 
 
# Incremental indexing summary: 
#   New documents:     0 
#   Updated documents: 0 
#   Skipped (unchanged): 7 
#   Skipped: ['account_opening_policy.pdf', 'complaint_handling_policy.pdf', 'credit_card_policy.pdf', 'customer_faq.pdf', 'loan_policy.pdf', 'schedule_of_charges.pdf', 'spytm_Scanned.pdf'] 
 
# All documents already indexed — loading existing store 
# D:\Agentic AI\AI-powered bank customer-support assistant\AI-powered-bank-customer-support-assistant\retrieval\embedder.py:15: LangChainDeprecationWarning: The class `HuggingFaceEmbeddings` was deprecated in LangChain 0.2.2 and will be removed in 1.0. An updated version of the class exists in the `langchain-huggingface package and should be used instead. To use it run `pip install -U `langchain-huggingface` and import as `from `langchain_huggingface import HuggingFaceEmbeddings``. 
#   return HuggingFaceEmbeddings( 
# Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads. 
# Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 103/103 [00:00<00:00, 5511.50it/s] 
# D:\Agentic AI\AI-powered bank customer-support assistant\AI-powered-bank-customer-support-assistant\retrieval\store.py:17: LangChainDeprecationWarning: The class `Chroma` was deprecated in LangChain 0.2.9 and will be removed in 1.0. An updated version of the class exists in the `langchain-chroma package and should be used instead. To use it run `pip install -U `langchain-chroma` and import as `from `langchain_chroma import Chroma``. 
#   return Chroma( 
# Total chunks in vector store: 66 
# Vector store ready 
# account_opening_policy.pdf → policy 
# complaint_handling_policy.pdf → policy 
# credit_card_policy.pdf → policy 
# customer_faq.pdf → faq 
# loan_policy.pdf → policy 
# schedule_of_charges.pdf → charges 
# spytm_Scanned.pdf → general 
# Recommended for policy: {'chunk_size': 600, 'overlap': 100} 
# Recursive chunking: 2 docs → 7 chunks 
# Recommended for policy: {'chunk_size': 600, 'overlap': 100} 
# Recursive chunking: 3 docs → 11 chunks 
# Recommended for policy: {'chunk_size': 600, 'overlap': 100} 
# Recursive chunking: 3 docs → 8 chunks 
# Recommended for faq: {'chunk_size': 400, 'overlap': 50} 
# Recursive chunking: 3 docs → 18 chunks 
# Recommended for policy: {'chunk_size': 600, 'overlap': 100} 
# Recursive chunking: 3 docs → 8 chunks 
# Recommended for charges: {'chunk_size': 300, 'overlap': 30} 
# Recursive chunking: 2 docs → 13 chunks 
# Recommended for general: {'chunk_size': 500, 'overlap': 50} 
# Recursive chunking: 1 docs → 1 chunks 
 
# Chunks prepared for BM25: 66 
 
# ============================================================ 
# STEP 6 — SETTING UP LLM 
# ============================================================ 
# LLM configured 
 
# ============================================================ 
# STEP 7 — CREATING RAG PROMPT 
# ============================================================ 
# RAG prompt created 
 
# ============================================================ 
# STEP 9 — BUILDING RAG CHAIN 
# ============================================================ 
# RAG chain ready 
 
# ============================================================ 
# DEBUGGING: What is the minimum balance required for a savings account? 
# ============================================================ 
 
# --- STEP 1: RETRIEVAL --- 
 
# ================================================== 
# SMART RETRIEVAL 
# Strategy: ensemble | k=4 | reorder=True 
# Question: What is the minimum balance required for a savings account?... 
# ================================================== 
# Detected relevant document: account_opening_policy.pdf (score: 2) 
 
# Reordered 4 chunks — best at start and end 
 
# Final context: 2493 chars from 4 chunks 
# Sources: ['account_opening_policy.pdf', 'account_opening_policy.pdf', 'account_opening_policy.pdf', 'account_opening_policy.pdf'] 
 
# Retrieved 4 chunks: 
 
#   Chunk 1: 
#   Source: account_opening_policy.pdf 
#   Page: 1 
#   Preview: Foreign Currency Account: USD 100 minimum balance  
  
# Below minimum balance charge: Rs. 500 per month 
 
#   Chunk 2: 
#   Source: account_opening_policy.pdf 
#   Page: 2 
#   Preview: Debit card issued within 7 working days  
# Online banking activated within 24 hours  
# Cheque book issue 
 
#   Chunk 3: 
#   Source: account_opening_policy.pdf 
#   Page: 1 
#   Preview: ABC BANK — ACCOUNT OPENING POLICY  
# Version 1.0 | Effective Date: January 2024  
  
# SECTION 1 — TYPES O 
 
#   Chunk 4: 
#   Source: account_opening_policy.pdf 
#   Page: 1 
#   Preview: 1.2 Business Accounts  
# Current Account for Business — unlimited transactions, no profit  
# Business Sa 
 
# --- STEP 2: AUGMENTATION --- 
# Context length: 2493 characters 
# Context preview: 
# [Document: account_opening_policy.pdf | Page: 1] 
# Foreign Currency Account: USD 100 minimum balance  
  
# Below minimum balance charge: Rs. 500 per month for savings  
# Below minimum balance charge: Rs. 1,000 per month for current  
  
# SECTION 3 — PROFIT RATES  
  
# Regular Savings Account: 5% per annum  
# Premium Savings Account: 7% per annum  
# Basic Banking Account: 0% profit  
# Profit credited: monthly on last... 
 
# --- STEP 3: GENERATION --- 
 
# ================================================== 
# SMART RETRIEVAL 
# Strategy: ensemble | k=4 | reorder=True 
# Question: What is the minimum balance required for a savings account?... 
# ================================================== 
# Detected relevant document: account_opening_policy.pdf (score: 2) 
 
# Reordered 4 chunks — best at start and end 
 
# Final context: 2493 chars from 4 chunks 
# Sources: ['account_opening_policy.pdf', 'account_opening_policy.pdf', 'account_opening_policy.pdf', 'account_opening_policy.pdf'] 
 
# Answer: 
# The minimum balance required for a Regular Savings Account is **Rs. 10,000**. 
 
# ============================================================ 
# DEBUGGING: Which documents do I need to open a bank account? 
# ============================================================ 
 
# --- STEP 1: RETRIEVAL --- 
 
# ================================================== 
# SMART RETRIEVAL 
# Strategy: ensemble | k=4 | reorder=True 
# Question: Which documents do I need to open a bank account?... 
# ================================================== 
# Detected relevant document: account_opening_policy.pdf (score: 2) 
 
# Reordered 4 chunks — best at start and end 
 
# Final context: 2132 chars from 4 chunks 
# Sources: ['account_opening_policy.pdf', 'account_opening_policy.pdf', 'account_opening_policy.pdf', 'account_opening_policy.pdf'] 
 
# Retrieved 4 chunks: 
 
#   Chunk 1: 
#   Source: account_opening_policy.pdf 
#   Page: 1 
#   Preview: 1.2 Business Accounts  
# Current Account for Business — unlimited transactions, no profit  
# Business Sa 
 
#   Chunk 2: 
#   Source: account_opening_policy.pdf 
#   Page: 2 
#   Preview: Minor original B-Form  
# Parent or guardian original CNIC  
# Proof of relationship — birth certificate  
 
 
#   Chunk 3: 
#   Source: account_opening_policy.pdf 
#   Page: 1 
#   Preview: Original CNIC — mandatory  
# Utility bill not older than 3 months — proof of address  
# Passport size ph 
 
#   Chunk 4: 
#   Source: account_opening_policy.pdf 
#   Page: 1 
#   Preview: ABC BANK — ACCOUNT OPENING POLICY  
# Version 1.0 | Effective Date: January 2024  
  
# SECTION 1 — TYPES O 
 
# --- STEP 2: AUGMENTATION --- 
# Context length: 2132 characters 
# Context preview: 
# [Document: account_opening_policy.pdf | Page: 1] 
# 1.2 Business Accounts  
# Current Account for Business — unlimited transactions, no profit  
# Business Savings Account — limited transactions with profit  
# Corporate Account — for registered companies  
  
# SECTION 2 — MINIMUM BALANCE REQUIREMENTS  
  
# Basic Banking Account: Rs. 0 minimum balance  
# Regular Savings Account: Rs. 10,000 minimum balance  
# Premium Sa... 
 
# --- STEP 3: GENERATION --- 
 
# ================================================== 
# SMART RETRIEVAL 
# Strategy: ensemble | k=4 | reorder=True 
# Question: Which documents do I need to open a bank account?... 
# ================================================== 
# Detected relevant document: account_opening_policy.pdf (score: 2) 
 
# Reordered 4 chunks — best at start and end 
 
# Final context: 2132 chars from 4 chunks 
# Sources: ['account_opening_policy.pdf', 'account_opening_policy.pdf', 'account_opening_policy.pdf', 'account_opening_policy.pdf'] 
 
# Answer: 
# To open an account at ABC Bank you’ll need to provide the following documents: 
 
# | Account type | Required documents | 
# |--------------|--------------------| 
# | **All accounts (individual, savings, current, foreign‑currency, etc.)** | • Original CNIC (mandatory)  <br>• Utility bill (not older than 3 months) – proof of address  <br>• Two passport‑size photographs  <br>• Source of income declaration form (mandatory)  <br>• Next‑of‑kin details (mandatory) | 
# | **Minor accounts** | • Minor’s original B‑Form  <br>• Parent or guardian’s original CNIC  <br>• Proof of relationship (birth certificate)  <br>• Guardian’s utility bill | 
# | **Business accounts** | • Business owner’s CNIC  <br>• Business registration certificate  <br>• NTN certificate  <br>• Board resolution for authorized signatories  <br>• Partnership deed (if applicable)  <br>• Memorandum and articles of association (for companies) | 
 
# Make sure all documents are original and meet the specified requirements before submitting them to the bank. 
 
# ============================================================ 
# DEBUGGING: How much is the IBFT transfer charge? 
# ============================================================ 
 
# --- STEP 1: RETRIEVAL --- 
 
# ================================================== 
# SMART RETRIEVAL 
# Strategy: ensemble | k=4 | reorder=True 
# Question: How much is the IBFT transfer charge?... 
# ================================================== 
# Detected relevant document: schedule_of_charges.pdf (score: 3) 
 
# Reordered 4 chunks — best at start and end 
 
# Final context: 1314 chars from 4 chunks 
# Sources: ['schedule_of_charges.pdf', 'schedule_of_charges.pdf', 'schedule_of_charges.pdf', 'schedule_of_charges.pdf'] 
 
# Retrieved 4 chunks: 
 
#   Chunk 1: 
#   Source: schedule_of_charges.pdf 
#   Page: 1 
#   Preview: Cheque return charges inward: Rs. 1,000 per cheque  
# Cheque return charges outward: Rs. 500 per chequ 
 
#   Chunk 2: 
#   Source: schedule_of_charges.pdf 
#   Page: 2 
#   Preview: Foreign currency conversion rate: SBP rate plus 3.5%  
  
# SECTION 4 — CARD CHARGES  
  
# Debit card issua 
 
#   Chunk 3: 
#   Source: schedule_of_charges.pdf 
#   Page: 2 
#   Preview: Profit and loss certificate: Rs. 500  
# Reference letter: Rs. 1,000  
# Swift message charges outward: Rs 
 
#   Chunk 4: 
#   Source: schedule_of_charges.pdf 
#   Page: 1 
#   Preview: IBFT above Rs. 500,000: Rs. 500 per transaction  
# Internal fund transfer within ABC Bank: Free unlimi 
 
# --- STEP 2: AUGMENTATION --- 
# Context length: 1314 characters 
# Context preview: 
# [Document: schedule_of_charges.pdf | Page: 1] 
# Cheque return charges inward: Rs. 1,000 per cheque  
# Cheque return charges outward: Rs. 500 per cheque  
# Stop payment instruction: Rs. 500  
  
# 2.3 Online and Digital Transactions  
# IBFT up to Rs. 25,000: Rs. 200 per transaction  
# IBFT Rs. 25,001 to Rs. 500,000: Rs. 300 per transaction 
 
# --- 
 
# [Document: schedule_of_charges.pdf | Page: 2] 
# Foreign currency conv... 
 
# --- STEP 3: GENERATION --- 
 
# ================================================== 
# SMART RETRIEVAL 
# Strategy: ensemble | k=4 | reorder=True 
# Question: How much is the IBFT transfer charge?... 
# ================================================== 
# Detected relevant document: schedule_of_charges.pdf (score: 3) 
 
# Reordered 4 chunks — best at start and end 
 
# Final context: 1314 chars from 4 chunks 
# Sources: ['schedule_of_charges.pdf', 'schedule_of_charges.pdf', 'schedule_of_charges.pdf', 'schedule_of_charges.pdf'] 
 
# Answer: 
# IBFT transfer charges vary by the amount transferred: 
 
# | Transfer amount | Charge per transaction | 
# |-----------------|------------------------| 
# | Up to Rs. 25,000 | Rs. 200 | 
# | Rs. 25,001 – Rs. 500,000 | Rs. 300 | 
# | Above Rs. 500,000 | Rs. 500 | 
 
# ============================================================ 
# DEBUGGING: What is the procedure for filing a complaint against the bank? 
# ============================================================ 
 
# --- STEP 1: RETRIEVAL --- 
 
# ================================================== 
# SMART RETRIEVAL 
# Strategy: ensemble | k=4 | reorder=True 
# Question: What is the procedure for filing a complaint against the ban... 
# ================================================== 
# Detected relevant document: complaint_handling_policy.pdf (score: 1) 
 
# Reordered 4 chunks — best at start and end 
 
# Final context: 2051 chars from 4 chunks 
# Sources: ['complaint_handling_policy.pdf', 'complaint_handling_policy.pdf', 'complaint_handling_policy.pdf', 'complaint_handling_policy.pdf'] 
 
# Retrieved 4 chunks: 
 
#   Chunk 1: 
#   Source: complaint_handling_policy.pdf 
#   Page: 3 
#   Preview: Right to escalate if not satisfied with resolution  
# Right to approach Banking Mohtasib if internal r 
 
#   Chunk 2: 
#   Source: complaint_handling_policy.pdf 
#   Page: 1 
#   Preview: CNIC number  
# Account number or card number  
# Contact number and email address  
# Clear description of t 
 
#   Chunk 3: 
#   Source: complaint_handling_policy.pdf 
#   Page: 1 
#   Preview: ABC BANK — COMPLAINT HANDLING POLICY  
# Version 1.0 | Effective Date: January 2024  
  
# SECTION 1 — POLI 
 
#   Chunk 4: 
#   Source: complaint_handling_policy.pdf 
#   Page: 1 
#   Preview: 2.1 Available Channels  
# Phone: Call 0800-ABCBANK available 24 hours 7 days a week  
# Branch: Visit any 
 
# --- STEP 2: AUGMENTATION --- 
# Context length: 2051 characters 
# Context preview: 
# [Document: complaint_handling_policy.pdf | Page: 3] 
# Right to escalate if not satisfied with resolution  
# Right to approach Banking Mohtasib if internal resolution fails  
# Right to confidentiality of complaint details 
 
# --- 
 
# [Document: complaint_handling_policy.pdf | Page: 1] 
# CNIC number  
# Account number or card number  
# Contact number and email address  
# Clear description of the complaint  
# Date and loca... 
 
# --- STEP 3: GENERATION --- 
 
# ================================================== 
# SMART RETRIEVAL 
# Strategy: ensemble | k=4 | reorder=True 
# Question: What is the procedure for filing a complaint against the ban... 
# ================================================== 
# Detected relevant document: complaint_handling_policy.pdf (score: 1) 
 
# Reordered 4 chunks — best at start and end 
 
# Final context: 2051 chars from 4 chunks 
# Sources: ['complaint_handling_policy.pdf', 'complaint_handling_policy.pdf', 'complaint_handling_policy.pdf', 'complaint_handling_policy.pdf'] 
 
# Answer: 
# **Procedure for filing a complaint against ABC Bank** 
 
# 1. **Choose a filing channel**   
#    - **Phone**: Call 0800‑ABCBANK (available 24 h/7 d).   
#    - **Branch**: Visit any ABC Bank branch and speak to a Customer Service Officer.   
#    - **Email**: Send an email to complaints@abcbank.com (response within 24 h).   
#    - **Mobile App**: Open the ABC Bank app → Help section → File Complaint.   
#    - **Website**: Go to abcbank.com/complaints and fill out the online form.   
#    - **Written**: Mail a letter to the Complaints Department, Head Office, Karachi. 
 
# 2. **Provide the required information**   
#    - Full name as per CNIC   
#    - CNIC number   
#    - Account number or card number   
#    - Contact number and email address   
#    - Clear description of the complaint   
#    - Date and location of the incident   
#    - Any reference numbers related to the issue   
#    - Supporting documents (if available) 
 
# 3. **After registration**   
#    - A complaint reference number is issued immediately.   
#    - You will receive an SMS confirmation within 1 hour and an email confirmation within 2 hours.   
#    - Use this reference number for all follow‑up inquiries. 
 
# Follow these steps to file a complaint and ensure it is processed efficiently. 
 
# ============================================================ 
# DEBUGGING: How much does a RAAST transfer cost? 
# ============================================================ 
 
# --- STEP 1: RETRIEVAL --- 
 
# ================================================== 
# SMART RETRIEVAL 
# Strategy: ensemble | k=4 | reorder=True 
# Question: How much does a RAAST transfer cost?... 
# ================================================== 
# Detected relevant document: schedule_of_charges.pdf (score: 3) 
 
# Reordered 4 chunks — best at start and end 
 
# Final context: 1220 chars from 4 chunks 
# Sources: ['schedule_of_charges.pdf', 'schedule_of_charges.pdf', 'schedule_of_charges.pdf', 'schedule_of_charges.pdf'] 
 
# Retrieved 4 chunks: 
 
#   Chunk 1: 
#   Source: schedule_of_charges.pdf 
#   Page: 1 
#   Preview: Mini statement at own ATM: Free up to 2 per month  
# Mini statement above 2: Rs. 25 per transaction  
  
 
#   Chunk 2: 
#   Source: schedule_of_charges.pdf 
#   Page: 2 
#   Preview: Profit and loss certificate: Rs. 500  
# Reference letter: Rs. 1,000  
# Swift message charges outward: Rs 
 
#   Chunk 3: 
#   Source: schedule_of_charges.pdf 
#   Page: 2 
#   Preview: Balance inquiry at other bank ATM: Rs. 18 per transaction  
# Maximum 5 free transactions per month for 
 
#   Chunk 4: 
#   Source: schedule_of_charges.pdf 
#   Page: 1 
#   Preview: IBFT above Rs. 500,000: Rs. 500 per transaction  
# Internal fund transfer within ABC Bank: Free unlimi 
 
# --- STEP 2: AUGMENTATION --- 
# Context length: 1220 characters 
# Context preview: 
# [Document: schedule_of_charges.pdf | Page: 1] 
# Mini statement at own ATM: Free up to 2 per month  
# Mini statement above 2: Rs. 25 per transaction  
  
# 3.2 Other Bank ATM in Pakistan  
# Cash withdrawal at other bank ATM: Rs. 25 per transaction 
 
# --- 
 
# [Document: schedule_of_charges.pdf | Page: 2] 
# Profit and loss certificate: Rs. 500  
# Reference letter: Rs. 1,000  
# Swift message charges outward: Rs. 1,500  
  
# ... 
 
# --- STEP 3: GENERATION --- 
 
# ================================================== 
# SMART RETRIEVAL 
# Strategy: ensemble | k=4 | reorder=True 
# Question: How much does a RAAST transfer cost?... 
# ================================================== 
# Detected relevant document: schedule_of_charges.pdf (score: 3) 
 
# Reordered 4 chunks — best at start and end 
 
# Final context: 1220 chars from 4 chunks 
# Sources: ['schedule_of_charges.pdf', 'schedule_of_charges.pdf', 'schedule_of_charges.pdf', 'schedule_of_charges.pdf'] 
 
# Answer: 
# RAAST transfer: Free unlimited. 
# (venv) PS D:\Agentic AI\AI-powered bank customer-support assistant\AI-powered-bank-customer-support-assistant>  these are the output i got after apply different starteget 