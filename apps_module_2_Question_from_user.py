import streamlit as st
#from transformers import pipeline

#https://stackoverflow.com/questions/56611698/pandas-how-to-read-csv-file-from-google-drive-public
#https://stackoverflow.com/questions/18885175/read-a-zipped-file-as-a-pandas-dataframe 

import pandas as pd
url='https://drive.google.com/file/d/1jRImDe_etDIGjB9rBYXxsdcPdmqa9x8R/view?usp=sharing'
#url='https://drive.google.com/file/d/1DlpbMAqIB50aJVyMRES_J_CnxYdRXH-p/view?usp=sharing'
df = pd.read_csv('https://drive.usercontent.google.com/download?id={}&export=download&authuser=0&confirm=t'.format(url.split('/')[-2]), compression='zip', sep='##', names=['text', 'key', 'period' , 'title'])

import os

from transformers import pipeline

import pandas as pd



opts = df['key'].tolist()
opts_c = []

for op in opts:
    if op not in opts_c:
        opts_c.append(op)

opts = opts_c 
opts.sort()

st.set_page_config(
    page_title="Text Generation App", page_icon=":pencil2:", layout="wide"
)

# Add a title and a description
st.title("Module_1_Input_your_query_for_the_Christian_thinker")
st.write(
    "This application generates question answer based on the topic choosen on the author's text of interest."
)

# Initialize the Hugging Face model
#generator = pipeline("text-generation", model="gpt2")

# Create a sidebar for input parameters
st.sidebar.title("User input")
opts_1 = [None, 'John_Chrysostom_Early_Church', 'Augustine_Early_Church', 'Thomas_Aquinas_Medieval_Scholasticism', 'John_Calvin_Protestant_Reformer', 'Charles_Spurgeon_Evangelicalism']

    # Create a sidebar for input parameters
st.sidebar.title("User input")
option1 = st.sidebar.selectbox(
        "Select author of interest",
        opts_1,
)


input_text = st.sidebar.text_input(label="User query", value="")
option = st.sidebar.selectbox(
    "Select topic of interest?",
    opts,
)

generate_button = st.sidebar.button("Generate")




url = ''
if generate_button:
    if option1 == 'John_Chrysostom_Early_Church':
        url='https://drive.google.com/file/d/1dZuiBxM4L7CQM4tSoneaZD7URC27zPNz/view?usp=sharing'
    if option1 == 'Augustine_Early_Church':
        url='https://drive.google.com/file/d/1hgqLQEAQSdLBWK0SXbCMrq2y_pho4Unj/view?usp=sharing'
    if option1 == 'Thomas_Aquinas_Medieval_Scholasticism':
        url='https://drive.google.com/file/d/11ueG9OJ6kAibEK2WO9Vm4sM51Gi5S7c1/view?usp=sharing'
    if option1 == 'John_Calvin_Protestant_Reformer':
        url='https://drive.google.com/file/d/198J8IDnxaFNw95weam_7RzD-HdtarMwy/view?usp=sharing'
    if option1 == 'Charles_Spurgeon_Evangelicalism':
        url='https://drive.google.com/file/d/1SnRbG9NZMWwNbJkl49aDVLVls7YEIL4n/view?usp=sharing'
    

    df = pd.read_csv('https://drive.usercontent.google.com/download?id={}&export=download&authuser=0&confirm=t'.format(url.split('/')[-2]), compression='zip', sep='##', names=['text', 'key', 'period' , 'title'])

    
    #hf_token ="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"

    #os.environ["HUGGINGFACEHUB_API_TOKEN"]= hf_token # replace hf_token with your HuggingFace API-token 
                
    df1 = df.loc[df['key'] == option]

    print(option)

    print(df1)

    hi = df1['text'].tolist()         

    

    arts = ''
    c_l = []
    collected = []


    for h in hi:
            cf1 = str(h).count(option)
            cf2 = str(h).count(str(option).lower())
            cf3 = cf1 + cf2
            if cf3 > 0:
                
        
                arts += h + '\n'
                c_l.append(h)


    # Set page title and description

    # Create a button to generate text


    # Generate and display text when the button is clicked
    try:
        os.remove('modul1_inter1.txt')
    except:
        pass
    try:
        os.remove('modul1_inter2.txt')
    except:
        pass
    try:
        os.remove('generated_summary_module36.txt')
    except:
        pass

    import time

    m_str = ''
    C = 0
    main_res = ''

    for c in c_l:

        ny1 = open('modul1_inter2.txt', 'a+')
        ny1.write(str(c) + '\n')
        ny1.close()
        print('here')
        
        
    C = 1  

    if C == 1: 
        
        if C == 1:
        
            from langchain.document_loaders import TextLoader
            loader = TextLoader('modul1_inter2.txt')
            documents = loader.load()
            
                        
            from langchain.text_splitter import CharacterTextSplitter
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            docs = text_splitter.split_documents(documents)
            
            # Embeddings
            from langchain.embeddings import HuggingFaceEmbeddings
            embeddings = HuggingFaceEmbeddings()
            from langchain.vectorstores import FAISS
            db = FAISS.from_documents(docs, embeddings)
            query = option
            docs = db.similarity_search(query)
            
            from langchain.chains.question_answering import load_qa_chain
            from langchain import HuggingFaceHub
            from langchain.output_parsers.regex import RegexParser
            
            hf_token2 ="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"
            import os
        
            os.environ["HUGGINGFACEHUB_API_TOKEN"]= hf_token2 # replace hf_token with your HuggingFace API-token 
        
            #llm=HuggingFaceHub(repo_id="togethercomputer/RedPajama-INCITE-Chat-3B-v1", model_kwargs={"temperature":0.8, "max_length":512})
            llm=HuggingFaceHub(repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1", model_kwargs={"temperature":0.8, "max_new_tokens":1028})
            
            from langchain.prompts import PromptTemplate
            #Generate a questions the topic marriage for which answers can be found strictly in the given passage below
        
            prompt = PromptTemplate.from_template(
            """
            Strictly based on the text given below answer the question '"""+str(input_text)+"""' by referencing passages from the text:
            
            {context}
            
            
            Question: {question}
            Helpful Answer:
            """)
            
            
            
            #documents = 'But let us see in what sense Marriage is honorable in all and the bed undefiled. Because  it preserves the believer in chastity. Here he also alludes to the Jews, because they accounted the woman after childbirth polluted: and whosoever comes from the bed, it is said, is not clean. Those things are not polluted which arise from nature, O ungrateful and senseless Jew, but those which arise from choice. For if marriage is honorable and pure, why forsooth dost thou think that one is even polluted by it?'
            query = ""
            
            
            chain = load_qa_chain(llm, chain_type="stuff", 
                prompt=prompt,
                # this is the default values and can be modified/omitted
                document_variable_name="context",)
            
            reps = chain.run({"input_documents": docs, "question": query})
            #print('1')
            #print(reps)
            #time.sleep(10000)
            
            

            dv = str(reps).split('Helpful Answer:')
            tgh = str(dv[-1]).split('\n')

            

            for th in tgh:
                bkl = open('generated_summary_module36.txt', 'a+')
                bkl.write(str(th).strip() + '\n')
                bkl.close()
            C = 0
            os.remove('modul1_inter2.txt')



        if C == 2:
        
            from langchain.document_loaders import TextLoader
            loader = TextLoader('modul1_inter1.txt')
            documents = loader.load()
            
            
            from langchain.text_splitter import CharacterTextSplitter
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            docs = text_splitter.split_documents(documents)
            
            # Embeddings
            from langchain.embeddings import HuggingFaceEmbeddings
            embeddings = HuggingFaceEmbeddings()
            from langchain.vectorstores import FAISS
            db = FAISS.from_documents(docs, embeddings)
            query = option
            docs = db.similarity_search(query)
            
            from langchain.chains.question_answering import load_qa_chain
            from langchain import HuggingFaceHub
            from langchain.output_parsers.regex import RegexParser
            
            hf_token2 ="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"
            os.environ["HUGGINGFACEHUB_API_TOKEN"]= hf_token2 # replace hf_token with your HuggingFace API-token 
        
            #llm=HuggingFaceHub(repo_id="togethercomputer/RedPajama-INCITE-Chat-3B-v1", model_kwargs={"temperature":0.8, "max_length":512})
            llm=HuggingFaceHub(repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1", model_kwargs={"temperature":0.8, "max_new_tokens":1028})
            
            from langchain.prompts import PromptTemplate
            #Generate a questions the topic marriage for which answers can be found strictly in the given passage below
        
            prompt = PromptTemplate.from_template(
            """
            Strictly based on the text given below answer the question '"""+str(input_text)+"""' by referencing passages from the text:
            
            {context}
            
            
            Question: {question}
            Helpful Answer:
            """)
            
            
            
            #documents = 'But let us see in what sense Marriage is honorable in all and the bed undefiled. Because  it preserves the believer in chastity. Here he also alludes to the Jews, because they accounted the woman after childbirth polluted: and whosoever comes from the bed, it is said, is not clean. Those things are not polluted which arise from nature, O ungrateful and senseless Jew, but those which arise from choice. For if marriage is honorable and pure, why forsooth dost thou think that one is even polluted by it?'
            query = ""
            
            
            chain = load_qa_chain(llm, chain_type="stuff", 
                prompt=prompt,
                # this is the default values and can be modified/omitted
                document_variable_name="context",)
            
            reps = chain.run({"input_documents": docs, "question": query})
            #print(reps)
            #time.sleep(10000)

            dv = str(reps).split('Helpful Answer:')
            tgh = str(dv[-1]).split('\n')

            main_res += tgh

            for th in tgh:
                bkl = open('generated_summary_module36.txt', 'a+')
                bkl.write(str(th).strip() + '\n')
                bkl.close()
            os.remove('modul1_inter1.txt')




    tio = open('generated_summary_module36.txt', 'r')
    output_text  = tio.readlines()
    tio.close()



    for th in output_text:
        main_res += str(th) 


            
            
        
    
            
        
    if generate_button:
        with st.spinner("Generating text..."):
            output_text_real = main_res
        st.subheader("Generated Text:")
        st.write(output_text_real)
