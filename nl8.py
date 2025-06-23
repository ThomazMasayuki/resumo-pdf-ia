import os
import re
import streamlit as st
from PIL import Image
import pytesseract
from transformers import pipeline
import fitz  # PyMuPDF

# === Limpeza e correção de texto extraído do OCR ===
def limpar_texto(texto):
    texto = re.sub(r'-\s*\n', '', texto)
    texto = re.sub(r'\n+', ' ', texto)
    texto = re.sub(r'\s{2,}', ' ', texto)
    texto = re.sub(r'[^\w\s,.!?;:–-]', '', texto)
    return texto.strip()

# === Função para extrair texto de PDF (em bytes) usando PyMuPDF ===
def extrair_texto_de_pdf(uploaded_file):
    texto_total = ""
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    for i, pagina in enumerate(doc):
        imagem = pagina.get_pixmap()
        img = Image.frombytes("RGB", [imagem.width, imagem.height], imagem.samples)
        texto = pytesseract.image_to_string(img, lang='por')
        texto_total += f"\n[Página {i+1}]\n{texto}"
    return texto_total

# === Carregar pipeline de sumarização ===
@st.cache_resource(show_spinner=False)
def carregar_pipeline():
    return pipeline("summarization", model="facebook/bart-large-cnn")

# === Sumarizar por blocos com modelo neural ===
def resumir_com_pipeline(texto, pipe, tamanho_bloco=1500):
    partes = [texto[i:i+tamanho_bloco] for i in range(0, len(texto), tamanho_bloco)]
    resumos = []
    for parte in partes:
        try:
            resultado = pipe(parte, max_length=200, min_length=80, do_sample=False)
            resumo_bruto = resultado[0]['summary_text']
            resumos.append(resumo_bruto)
        except Exception as e:
            resumos.append(f"[Erro ao resumir trecho: {e}]")
    return "\n\n".join(resumos)

# === Interface Streamlit ===
st.title("📄 Resumo de PDF")
uploaded_file = st.file_uploader("Selecione um arquivo PDF", type=["pdf"])

if uploaded_file:
    st.info("Extraindo e processando o texto... Isso pode levar alguns segundos.")
    texto_extraido = extrair_texto_de_pdf(uploaded_file)
    texto_limpo = limpar_texto(texto_extraido)
    pipe = carregar_pipeline()
    resumo = resumir_com_pipeline(texto_limpo, pipe)
    st.success("Resumo gerado com sucesso!")
    st.text_area("Resumo Interpretativo do Documento", resumo, height=500)