from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
from src.template_prompt import *

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY




def file_processing(file_path):
    # Load data from PDF
    loader = PyPDFLoader(file_path)
    data = loader.load()

    question_gen = ''.join([page.page_content for page in data])

    # Use RecursiveCharacterTextSplitter instead of TokenTextSplitter
    splitter_ques_gen = RecursiveCharacterTextSplitter(
        chunk_size=10000, 
        chunk_overlap=200
    )
    chunks_ques_gen = splitter_ques_gen.split_text(question_gen)
    document_ques_gen = [Document(page_content=t) for t in chunks_ques_gen]

    splitter_ans_gen = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100
    )
    document_answer_gen = splitter_ans_gen.split_documents(document_ques_gen)

    return document_ques_gen, document_answer_gen



def llm_pipeline(file_path):
    document_ques_gen, document_answer_gen = file_processing(file_path)

    # Use Gemini Pro instead of OpenAI's ChatOpenAI
    llm_ques_gen_pipeline = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.3,
        convert_system_message_to_human=True
    )

    PROMPT_QUESTIONS = PromptTemplate(template=prompt_template, input_variables=["text"])

    REFINE_PROMPT_QUESTIONS = PromptTemplate(
        input_variables=["existing_answer", "text"],
        template=refine_template,
    )

    # Load summarize chain with Gemini
    ques_gen_chain = load_summarize_chain(
        llm=llm_ques_gen_pipeline,
        chain_type="refine",
        verbose=True,
        question_prompt=PROMPT_QUESTIONS,
        refine_prompt=REFINE_PROMPT_QUESTIONS
    )

    ques = ques_gen_chain.run(document_ques_gen)

    # Use Gemini's embedding model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Create FAISS vector store
    vector_store = FAISS.from_documents(document_answer_gen, embeddings)

    # Use Gemini Pro for answer generation
    llm_answer_gen = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.1,
        convert_system_message_to_human=True
    )

    # Process the generated questions
    ques_list = ques.split("\n")
    filtered_ques_list = [element for element in ques_list if element.endswith('?') or element.endswith('.')]

    # Retrieval QA with Gemini
    answer_generation_chain = RetrievalQA.from_chain_type(
        llm=llm_answer_gen,
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )

    return answer_generation_chain, filtered_ques_list
