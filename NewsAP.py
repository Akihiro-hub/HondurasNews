# app.py - Version simplificada para debugging
import streamlit as st
import feedparser
import urllib.parse as up
from collections import Counter
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import re
import html
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from unidecode import unidecode
from datetime import date, datetime
import folium
# from streamlit_folium import st_folium  # Eliminado - no se usa
from wordcloud import WordCloud
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob

# Secretsからパスワードを取得
PASSWORD = st.secrets["PASSWORD"]

# Secretsからパスワードを取得
PASSWORD = st.secrets["PASSWORD"]

# パスワード認証の処理
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "login_attempts" not in st.session_state:
    st.session_state.login_attempts = 0

def verificar_contraseña():
    contraseña_ingresada = st.text_input("Introduce la contraseña:", type="password")

    if st.button("Iniciar sesión"):
        if st.session_state.login_attempts >= 3:
            st.error("Has superado el número máximo de intentos. Acceso bloqueado.")
        elif contraseña_ingresada == PASSWORD:  # Secretsから取得したパスワードで認証
            st.session_state.authenticated = True
            st.success("¡Autenticación exitosa! Marque otra vez el botón 'Iniciar sesión'.")
        else:
            st.session_state.login_attempts += 1
            intentos_restantes = 3 - st.session_state.login_attempts
            st.error(f"Contraseña incorrecta. Te quedan {intentos_restantes} intento(s).")
        
        if st.session_state.login_attempts >= 3:
            st.error("Acceso bloqueado. Intenta más tarde.")

if st.session_state.authenticated:
    # 認証成功後に表示されるメインコンテンツ
    # Configuración de la página
    st.set_page_config(
        page_title="Análisis de Noticias - Honduras", 
        page_icon="📰",
        layout="wide"
    )
    
    st.title("📰 Análisis de noticias Google en Honduras")
    st.caption("Busca palabras clave, analiza tendencias, mapas de ubicaciones y sentiment de noticias hondureñas")
    
    # ===== UI Principal para configuración =====
    st.subheader("⚙️ Configuración de Búsqueda")
    
    # Función para sanitizar entrada (XSS protection)
    def sanitize_input(text):
        if not text:
            return ""
        # Remover caracteres peligrosos
        text = re.sub(r'[<>"\'\&]', '', str(text))
        # Limitar longitud
        return text[:100].strip()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Palabras clave con protección XSS
        k1_raw = st.text_input("Palabra clave 1 (obligatoria)", value="Honduras")
        k1 = sanitize_input(k1_raw)
        
        k2_raw = st.text_input("Palabra clave 2 (opcional)", value="Turismo") 
        k2 = sanitize_input(k2_raw)
        
        k3_raw = st.text_input("Palabra clave 3 (opcional)", value="")
        k3 = sanitize_input(k3_raw)
    
    with col2:
        logic = st.radio("Lógica de búsqueda", ["AND (todas)", "OR (cualquiera)"], index=0)
        
        # Número de artículos (modificado: default 15, max 30)
        n_art = st.slider("Número de artículos a analizar", min_value=5, max_value=30, value=15, step=1)
    
    # Sidebar eliminado según solicitud del usuario
    
    # Configuración de parámetros
    hl, gl = "es-419", "HN"
    ceid = f"{gl}:{hl}"
    
    terms = [t.strip() for t in [k1, k2, k3] if t and t.strip()]
    if not terms:
        st.info("Introduce al menos una palabra clave en «Palabra clave 1».")
        st.stop()
    
    q = " + ".join(terms) if logic.startswith("AND") else " OR ".join(terms)
    base = "https://news.google.com/rss/search"
    params = {"q": q, "hl": hl, "gl": gl, "ceid": ceid}
    rss_url = base + "?" + up.urlencode(params, safe="+")
    
    # Coordenadas de Honduras
    HONDURAS_COORDS = {
        'Tegucigalpa': (14.0723, -87.1921),
        'San Pedro Sula': (15.5047, -88.0251),
        'La Ceiba': (15.7594, -86.7822),
        'Choloma': (15.6106, -87.9531),
        'Choluteca': (13.2969, -87.1914),
        'Comayagua': (14.4602, -87.6415),
        'Honduras': (14.0723, -87.1921),
    }
    
    # Ciudades principales de Honduras
    HONDURAS_CITIES = [
        'Tegucigalpa', 'San Pedro Sula', 'Choloma', 'La Ceiba', 'El Progreso',
        'Choluteca', 'Comayagua', 'Puerto Cortés', 'La Lima', 'Danlí',
        'Siguatepeque', 'Catacamas', 'Santa Rosa de Copán', 'Tela', 'Juticalpa',
        'Roatán', 'Utila', 'Guanaja', 'Copán', 'Trujillo', 'Omoa'
    ]
    
    # Meses en español
    MESES = ["enero","febrero","marzo","abril","mayo","junio",
             "julio","agosto","septiembre","octubre","noviembre","diciembre"]
    
    def fecha_es(d: date) -> str:
        return f"{d.day} de {MESES[d.month-1]} de {d.year}"
    
    # Stopwords expandidos
    STOPWORDS_ES = {
        "de","la","que","el","en","y","a","los","del","se","las","por","un","para","con","no",
        "una","su","al","lo","como","más","pero","sus","ya","o","este","sí","porque","esta",
        "entre","cuando","muy","sin","sobre","también","donde","desde","todo","todos","uno",
        "e","esto","hay","cada","ser","son","fue","han","ha","qué","cómo","cuándo","dónde",
        "rt","via","video","fotos","imagen","noticias","noticia","según","después","antes",
        "durante","mientras","además","aunque","embargo","través","partir","lugar","vez",
        "http","https","com","google","news","rss","href","target","_blank","font","color","nbsp",
        "amp","utm","html","site","link","article","articles","twitter","facebook","yahoo",
        "diario","diarios","expreso","expresos","digital","digitales",
        "periodico","periodicos","periódico","periódicos","medio","medios",
        # Ruido específico identificado por el usuario
        "canal","6f6f6f"
    }
    
    def clean_text(x: str) -> str:
        x = html.unescape(x or "")
        x = BeautifulSoup(x, "html.parser").get_text(separator=" ")
        x = re.sub(r"(https?://\S+|www\.\S+)", " ", x)
        return x
    
    def tokenize_clean(x: str):
        x = clean_text(x).lower()
        return re.findall(r"[a-záéíóúüñ]{3,}", x, re.IGNORECASE)
    
    def normalize_key(w: str) -> str:
        return unidecode(w.lower())
    
    def analyze_sentiment_simple(text):
        try:
            blob = TextBlob(text)
            sentiment = blob.sentiment
            polarity = sentiment.polarity
            subjectivity = sentiment.subjectivity
            
            if polarity > 0.1:
                classification = 'Positivo'
            elif polarity < -0.1:
                classification = 'Negativo'
            else:
                classification = 'Neutral'
                
            return {
                'polarity': polarity,
                'subjectivity': subjectivity,
                'classification': classification
            }
        except:
            return {
                'polarity': 0.0,
                'subjectivity': 0.0,
                'classification': 'Neutral'
            }
    
    def extract_locations_simple(texts):
        locations = Counter()
        combined_text = ' '.join(texts)
        
        for city in HONDURAS_CITIES:
            matches = len(re.findall(f'\\b{re.escape(city)}\\b', combined_text, re.IGNORECASE))
            if matches > 0:
                locations[city] = matches
        
        # Agregar Honduras como ubicación general
        honduras_mentions = len(re.findall(r'\bHonduras\b', combined_text, re.IGNORECASE))
        if honduras_mentions > 0:
            locations['Honduras'] = honduras_mentions
        
        return locations
    
    def create_simple_map(locations):
        try:
            if not locations:
                return None
            
            # Centro de Honduras
            center_lat, center_lon = 14.0723, -87.1921
            
            # Crear mapa
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=7,
                tiles='OpenStreetMap'
            )
            
            # Añadir marcadores
            max_mentions = max(locations.values()) if locations else 1
            
            for location, count in locations.items():
                if location in HONDURAS_COORDS:
                    lat, lon = HONDURAS_COORDS[location]
                    
                    # Tamaño del marcador
                    radius = min(20, max(8, (count / max_mentions) * 20))
                    
                    # Color basado en frecuencia
                    if count >= max_mentions * 0.7:
                        color = 'red'
                    elif count >= max_mentions * 0.4:
                        color = 'orange'
                    else:
                        color = 'green'
                    
                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=radius,
                        popup=f"{location}: {count} menciones",
                        tooltip=f"{location} ({count})",
                        color=color,
                        fill=True,
                        weight=2,
                        fillOpacity=0.7
                    ).add_to(m)
            
            return m
            
        except Exception as e:
            st.error(f"Error creando mapa: {e}")
            return None
    
    def cluster_topics_simple(texts, n_clusters=3):
        try:
            if len(texts) < 3:
                return [], None
            
            if len(texts) < n_clusters:
                n_clusters = max(1, len(texts) - 1)
            
            # Vectorización TF-IDF
            vectorizer = TfidfVectorizer(
                max_features=50,
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.8
            )
            
            # Filtrar textos válidos
            valid_texts = [text for text in texts if text and len(text.strip()) > 10]
            if len(valid_texts) < 2:
                return [], None
            
            tfidf_matrix = vectorizer.fit_transform(valid_texts)
            
            # Clustering
            if n_clusters > 1:
                try:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
                    cluster_labels = kmeans.fit_predict(tfidf_matrix)
                except:
                    # Fallback si KMeans falla
                    cluster_labels = np.zeros(len(valid_texts))
                    n_clusters = 1
            else:
                cluster_labels = np.zeros(len(valid_texts))
            
            # Obtener nombres de características
            feature_names = vectorizer.get_feature_names_out()
            
            # Analizar clusters
            clusters = []
            for i in range(n_clusters):
                cluster_indices = [j for j, label in enumerate(cluster_labels) if label == i]
                
                if not cluster_indices:
                    continue
                
                # Obtener top palabras del cluster de manera simple
                cluster_texts = [valid_texts[idx] for idx in cluster_indices]
                cluster_text = ' '.join(cluster_texts)
                tokens = tokenize_clean(cluster_text)
                token_counts = Counter(tokens)
                top_words = [word for word, count in token_counts.most_common(5) 
                            if word not in STOPWORDS_ES and len(word) > 2]
                
                cluster_name = " + ".join(top_words[:3]) if top_words else f"Tema {i+1}"
                
                clusters.append({
                    'name': cluster_name,
                    'keywords': top_words,
                    'articles': cluster_indices,
                    'size': len(cluster_indices)
                })
            
            # Crear wordcloud simple
            try:
                all_text = ' '.join(valid_texts)
                # Agregar las stopwords específicas de usuario
                extended_stopwords = STOPWORDS_ES.copy()
                extended_stopwords.update(['canal', '6f6f6f'])
                
                wordcloud = WordCloud(
                    width=800, 
                    height=400, 
                    background_color='white',
                    stopwords=extended_stopwords,
                    max_words=50,
                    colormap='viridis'
                ).generate(all_text)
                
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                plt.tight_layout()
                wordcloud_fig = fig
            except:
                wordcloud_fig = None
            
            return clusters, wordcloud_fig
            
        except Exception as e:
            st.error(f"Error en clustering: {e}")
            return [], None
    
    # Botón de análisis
    if st.button("🔍 Analizar Noticias", type="primary"):
        with st.spinner("Obteniendo y analizando noticias..."):
            try:
                feed = feedparser.parse(rss_url)
                
                if not feed.entries:
                    st.error("❌ No se encontraron noticias. Cambia las palabras clave o prueba con OR.")
                    st.stop()
    
                # Procesar artículos
                articles_data = []
                textos_completos = []
                
                for i, entry in enumerate(feed.entries[:n_art]):
                    titulo = getattr(entry, "title", "(sin título)")
                    resumen = getattr(entry, "summary", "")
                    enlace = getattr(entry, "link", "")
                    publicado_parsed = getattr(entry, "published_parsed", None)
                    
                    # Extraer fuente del enlace
                    source = "Desconocido"
                    if enlace:
                        try:
                            from urllib.parse import urlparse
                            domain = urlparse(enlace).netloc
                            source = domain.replace("www.", "").split(".")[0].title()
                        except:
                            pass
                    
                    fecha = None
                    if publicado_parsed:
                        fecha = datetime(publicado_parsed.tm_year, publicado_parsed.tm_mon, publicado_parsed.tm_mday)
                    
                    articles_data.append({
                        'titulo': titulo,
                        'resumen': resumen,
                        'enlace': enlace,
                        'fecha': fecha,
                        'fuente': source,
                        'texto_completo': f"{titulo} {resumen}"
                    })
                    
                    textos_completos.extend([titulo or "", resumen or ""])
    
                # ====== Mostrar artículos ======
                st.header("📰 Resultados de Noticias")
                
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.metric("Total de artículos", len(articles_data))
                with col2:
                    fuentes_unicas = len(set([art['fuente'] for art in articles_data]))
                    st.metric("Fuentes únicas", fuentes_unicas)
                with col3:
                    if articles_data and articles_data[0]['fecha']:
                        fecha_mas_reciente = max([art['fecha'] for art in articles_data if art['fecha']])
                        st.metric("Más reciente", fecha_mas_reciente.strftime("%d/%m"))
    
                # Mostrar artículos en expandibles
                for i, article in enumerate(articles_data, start=1):
                    with st.expander(f"📄 {i}. {article['titulo'][:100]}...", expanded=False):
                        if article['fecha']:
                            st.write(f"**📅 Fecha:** {fecha_es(article['fecha'].date())}")
                        st.write(f"**📺 Fuente:** {article['fuente']}")
                        if article['enlace']:
                            st.markdown(f"**🔗 [Ver artículo completo]({article['enlace']})**")
    
                # ====== Análisis de N-gramas ======
                st.header("🔤 Análisis de Frecuencias")
                
                search_norm = {normalize_key(t) for t in terms}
                def keep_token(tok: str) -> bool:
                    if tok in STOPWORDS_ES:
                        return False
                    if normalize_key(tok) in search_norm:
                        return False
                    return True
    
                texto_total = " ".join(textos_completos)
                tokens_all = tokenize_clean(texto_total)
                tokens = [t for t in tokens_all if keep_token(t)]
    
                # Mostrar unigramas
                cont_uni = Counter(tokens)
                top_uni = cont_uni.most_common(15)
                if top_uni:
                    df_uni = pd.DataFrame(data=top_uni, columns=['Palabra', 'Frecuencia'])
                    fig_uni = px.bar(df_uni, x='Frecuencia', y='Palabra', orientation='h',
                                   title="Top 15 Palabras más Frecuentes")
                    fig_uni.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_uni, use_container_width=True)
                else:
                    st.info("No hay suficientes palabras para el análisis de frecuencias.")
    
                # ====== NUEVAS FUNCIONALIDADES ======
                
                # 1. Análisis de Clustering de Temas (siempre activado)
                if len(articles_data) >= 3:
                    st.header("🧠 Análisis Automático de Temas")
                    try:
                        clusters, wordcloud_fig = cluster_topics_simple([art['texto_completo'] for art in articles_data])
                        
                        if clusters:
                            col1, col2 = st.columns([1, 1])
                            with col1:
                                st.subheader("Temas Identificados")
                                for i, cluster in enumerate(clusters):
                                    with st.expander(f"📋 Tema {i+1}: {cluster['name']}", expanded=True):
                                        st.write(f"**Artículos:** {len(cluster['articles'])}")
                                        st.write(f"**Palabras clave:** {', '.join(cluster['keywords'][:5])}")
                                        
                                        # Mostrar algunos títulos del cluster
                                        for j, article_idx in enumerate(cluster['articles'][:3]):
                                            if article_idx < len(articles_data):
                                                st.write(f"• {articles_data[article_idx]['titulo'][:60]}...")
    
                            with col2:
                                st.subheader("Nube de Palabras")
                                if wordcloud_fig:
                                    st.pyplot(wordcloud_fig)
                        else:
                            st.info("No se pudieron identificar temas específicos en los artículos.")
                            
                    except Exception as e:
                        st.warning(f"No se pudo realizar el análisis de temas: {str(e)}")
    
                # 2. Análisis de Sentimientos (siempre activado)
                st.header("💭 Análisis de Sentimientos")
                try:
                    sentiments = []
                    for article in articles_data:
                        sentiment = analyze_sentiment_simple(article['texto_completo'])
                        sentiments.append({
                            'titulo': article['titulo'][:50] + "...",
                            'polaridad': sentiment['polarity'],
                            'subjetividad': sentiment['subjectivity'],
                            'clasificacion': sentiment['classification']
                        })
                    
                    if sentiments:
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            # Distribución de sentimientos
                            sentiment_counts = Counter([s['clasificacion'] for s in sentiments])
                            df_sentiment = pd.DataFrame(data=list(sentiment_counts.items()), 
                                                      columns=['Sentimiento', 'Cantidad'])
                            
                            fig_sentiment = px.pie(df_sentiment, values='Cantidad', names='Sentimiento',
                                                 title="Distribución de Sentimientos",
                                                 color_discrete_map={'Positivo': 'green', 
                                                                   'Negativo': 'red', 
                                                                   'Neutral': 'gray'})
                            st.plotly_chart(fig_sentiment, use_container_width=True)
                        
                        with col2:
                            # Scatter plot de polaridad vs subjetividad
                            df_sent_scatter = pd.DataFrame(sentiments)
                            fig_scatter = px.scatter(df_sent_scatter, x='polaridad', y='subjetividad',
                                                   color='clasificacion', hover_data=['titulo'],
                                                   title="Polaridad vs Subjetividad",
                                                   color_discrete_map={'Positivo': 'green', 
                                                                     'Negativo': 'red', 
                                                                     'Neutral': 'gray'})
                            fig_scatter.update_layout(xaxis_title="Polaridad (-1 a 1)", 
                                                    yaxis_title="Subjetividad (0 a 1)")
                            st.plotly_chart(fig_scatter, use_container_width=True)
    
                        # Mostrar artículos más positivos y negativos
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            st.subheader("📰 Más Positivos")
                            positivos = sorted([s for s in sentiments if s['clasificacion'] == 'Positivo'], 
                                             key=lambda x: x['polaridad'], reverse=True)[:3]
                            for art in positivos:
                                st.write(f"• {art['titulo']} (Polaridad: {art['polaridad']:.2f})")
                        
                        with col2:
                            st.subheader("📰 Más Negativos")  
                            negativos = sorted([s for s in sentiments if s['clasificacion'] == 'Negativo'], 
                                             key=lambda x: x['polaridad'])[:3]
                            for art in negativos:
                                st.write(f"• {art['titulo']} (Polaridad: {art['polaridad']:.2f})")
                                
                except Exception as e:
                    st.warning(f"No se pudo realizar el análisis de sentimientos: {str(e)}")
    
            except Exception as e:
                st.error(f"Error general: {str(e)}")
                st.write("**Debug info:**")
                st.write(f"RSS URL: {rss_url}")
    
    else:
        st.info("👆 Configura las palabras clave y las opciones de análisis arriba, luego pulsa **Analizar Noticias** para ver los resultados completos.")

else:
    verificar_contraseña()
