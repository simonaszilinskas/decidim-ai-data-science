import gradio as gr
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import os
from mistralai import Mistral, UserMessage
from sklearn.cluster import KMeans
import plotly.express as px
import umap
from collections import defaultdict
import umap.umap_ as umap
import csv
from tempfile import NamedTemporaryFile

# Récupérer la clé API depuis les variables d'environnement
MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')
if not MISTRAL_API_KEY:
    raise EnvironmentError("La variable d'environnement MISTRAL_API_KEY n'est pas définie")

mistral_client = Mistral(api_key=MISTRAL_API_KEY)


def fetch_proposals(base_url, process_slug, page_size=100):
    """
    Fetch proposals from a Decidim participatory process using GraphQL with pagination
    """
    endpoint = f"{base_url}/api"
    all_proposals = []
    has_next_page = True
    cursor = None

    while has_next_page:
        query = """
        query GetProposalsForProcess($slug: String!, $first: Int, $after: String) {
          participatoryProcess(slug: $slug) {
            components {
              __typename
              ... on Proposals {
                proposals(first: $first, after: $after) {
                  nodes {
                    id
                    title {
                      translation(locale: "fr")
                    }
                    body {
                      translation(locale: "fr")
                    }
                    state
                    reference
                    endorsementsCount
                    createdAt
                    author {
                      name
                      nickname
                    }
                    category {
                      name {
                        translation(locale: "fr")
                      }
                    }
                    scope {
                      name {
                        translation(locale: "fr")
                      }
                    }
                    totalCommentsCount
                  }
                  pageInfo {
                    hasNextPage
                    endCursor
                  }
                }
              }
            }
          }
        }
        """

        variables = {
            "slug": process_slug,
            "first": page_size,
            "after": cursor
        }

        try:
            response = requests.post(
                endpoint,
                json={"query": query, "variables": variables},
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()

            for component in data['data']['participatoryProcess']['components']:
                if component['__typename'] == 'Proposals':
                    proposals_data = component['proposals']
                    all_proposals.extend(proposals_data['nodes'])
                    
                    page_info = proposals_data['pageInfo']
                    has_next_page = page_info['hasNextPage']
                    cursor = page_info['endCursor']
                    break
            
            if not has_next_page:
                break

        except Exception as e:
            print(f"Error fetching proposals: {str(e)}")
            break

    return {"data": {"participatoryProcess": {"components": [{"__typename": "Proposals", "proposals": {"nodes": all_proposals}}]}}}

def safe_get_translation(obj, default=None):
    """Safely get translation from nested dictionary structure"""
    if obj and isinstance(obj, dict):
        translation = obj.get('translation')
        return translation if translation is not None else default
    return default

def safe_get_nested(obj, *keys, default=None):
    """Safely get nested dictionary values"""
    try:
        for key in keys:
            if obj is None:
                return default
            obj = obj.get(key)
        return obj if obj is not None else default
    except (AttributeError, TypeError):
        return default

def process_proposals_data(data):
    """
    Convert GraphQL response into a pandas DataFrame with better error handling
    """
    proposals = []

    try:
        components = data['data']['participatoryProcess']['components']
        print(f"Found {len(components)} components")

        for component in components:
            print(f"Processing component type: {component['__typename']}")

            if component['__typename'] == 'Proposals' and component.get('proposals'):
                nodes = component['proposals'].get('nodes', [])
                if nodes:
                    print(f"Found {len(nodes)} proposals")

                    for node in nodes:
                        try:
                            title = safe_get_translation(safe_get_nested(node, 'title'))
                            body = safe_get_translation(safe_get_nested(node, 'body'))
                            
                            proposal = {
                                'id': node.get('id'),
                                'title': title,
                                'body': body,
                                'combined_text': f"{title} {body}",  # Add combined text field
                                'state': node.get('state'),
                                'reference': node.get('reference'),
                                'endorsements': node.get('endorsementsCount', 0),
                                'created_at': node.get('createdAt'),
                                'author_name': safe_get_nested(node, 'author', 'name'),
                                'author_nickname': safe_get_nested(node, 'author', 'nickname'),
                                'category': safe_get_translation(safe_get_nested(node, 'category', 'name')),
                                'scope': safe_get_translation(safe_get_nested(node, 'scope', 'name')),
                                'comments_count': node.get('totalCommentsCount', 0)
                            }
                            proposals.append(proposal)
                        except Exception as e:
                            print(f"Error processing proposal: {str(e)}")
                            continue

    except Exception as e:
        print(f"Error processing data: {str(e)}")
        return pd.DataFrame()

    # Create DataFrame
    df = pd.DataFrame(proposals)

    # Convert timestamp strings to datetime
    if 'created_at' in df.columns:
        df['created_at'] = pd.to_datetime(df['created_at'])

    return df

class MistralClient:
    """Singleton class for Mistral client to avoid multiple instantiations"""
    _instance = None
    _client = None

    @classmethod
    def initialize(cls):
        """Initialize the Mistral client with environment variable"""
        if cls._instance is None or cls._client is None:
            api_key = os.getenv('MISTRAL_API_KEY')
            if not api_key:
                raise EnvironmentError("La variable d'environnement MISTRAL_API_KEY n'est pas définie")
            cls._instance = cls()
            cls._client = Mistral(api_key=api_key)
        return cls._client

    @classmethod
    def get_instance(cls):
        """Get the Mistral client instance"""
        if cls._client is None:
            return cls.initialize()
        return cls._client


def get_embeddings_batch(texts: list[str], client: Mistral, batch_size: int = 32) -> list[list[float]]:
    """
    Get embeddings for a batch of texts with proper error handling and batching
    
    Args:
        texts: List of texts to embed
        client: Mistral client instance
        batch_size: Size of batches to process at once
        
    Returns:
        List of embeddings (each embedding is a list of floats)
    """
    all_embeddings = []
    
    # Process texts in batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        # Clean texts in batch
        batch = [text.replace("\n", " ").strip() for text in batch]
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.embeddings.create(
                    model="mistral-embed",
                    inputs=batch
                )
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Failed to get embeddings after {max_retries} attempts: {str(e)}")
                    # Add zero embeddings for failed batch
                    all_embeddings.extend([[0] * 1024] * len(batch))
                else:
                    import time
                    time.sleep(1)  # Wait before retrying
    
    return all_embeddings

def generate_embeddings(df: pd.DataFrame, client: Mistral) -> pd.DataFrame:
    """
    Generate embeddings for all proposals in the dataframe
    
    Args:
        df: DataFrame containing proposals
        client: Mistral client instance
        
    Returns:
        DataFrame with added embedding column
    """
    # Get all texts that need to be embedded
    texts = df['combined_text'].tolist()
    
    # Get embeddings in batches
    embeddings = get_embeddings_batch(texts, client)
    
    # Add embeddings to dataframe
    df['embedding'] = embeddings
    
    return df
def generate_categorization_analysis(results_df, client):
    """
    Generate analysis of categorization results
    """
    category_counts = results_df['categories'].value_counts()
    
    analysis_prompt = f"""Analyze these categorization results and provide insights:
    
    Category Distribution:
    {category_counts.to_string()}
    
    Provide a brief analysis of the distribution and any notable patterns."""

    messages = [UserMessage(content=analysis_prompt)]

    try:
        response = client.chat.complete(
            model="mistral-large-latest",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating analysis: {str(e)}"


def perform_clustering(df, n_clusters):
    matrix = np.vstack(df.embedding.values)
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42)
    kmeans.fit(matrix)
    df["Cluster"] = kmeans.labels_
    return df, matrix

def analyze_clusters(df, client):
    cluster_names = {}
    cluster_analyses = {}

    for cluster_id in sorted(df.Cluster.unique()):
        cluster_df = df[df.Cluster == cluster_id]
        sample_proposals = cluster_df.sample(min(5, len(cluster_df)))
        proposal_texts = "\n".join([
            f"Proposal {i+1}: {row['title']} - {row['body'][:200]}..."
            for i, row in enumerate(sample_proposals.to_dict('records'))
        ])

        messages = [
            UserMessage(content=f"Quel est le thème principal de ces propositions ? Donne un titre court (3-4 mots max) pour cet ensemble de propositions. Ne retourne rien d'autre - juste le titre.\n\n{proposal_texts}\n\nTitre:")
        ]

        try:
            response = client.chat.complete(
                model="mistral-large-latest",
                messages=messages,
                temperature=0,
            )
            theme = response.choices[0].message.content.strip()
            cluster_names[cluster_id] = theme

            cluster_analyses[cluster_id] = {
                'theme': theme,
                'size': len(cluster_df),
                'avg_endorsements': cluster_df.endorsements.mean(),
                'categories': cluster_df.category.value_counts().to_dict()
            }

        except Exception as e:
            cluster_names[cluster_id] = f"Cluster {cluster_id}"

    return cluster_names, cluster_analyses

def create_interactive_cluster_viz(df, matrix, cluster_names):
    """Create an interactive cluster visualization using UMAP and Plotly"""
    # Use UMAP for better cluster separation
    reducer = umap.UMAP(
        random_state=42,
        n_neighbors=15,
        min_dist=0.1,
        metric='cosine'  # Better for high-dimensional embeddings
    )
    embedding = reducer.fit_transform(matrix)
    
    # Create DataFrame for plotting
    plot_df = pd.DataFrame({
        'UMAP1': embedding[:, 0],
        'UMAP2': embedding[:, 1],
        'Cluster': [cluster_names[c] for c in df['Cluster']],
        'Title': df['title'],
        'Reference': df['reference']
    })
    
    # Create interactive scatter plot
    fig = px.scatter(
        plot_df,
        x='UMAP1',
        y='UMAP2',
        color='Cluster',
        hover_data=['Title', 'Reference'],
        title='Visualisation des clusters',
        template='plotly_white'
    )
    
    # Update layout for better visualization
    fig.update_traces(
        marker=dict(size=8),
        hovertemplate="<br>".join([
            "Cluster: %{customdata[0]}",
            "Title: %{customdata[1]}",
            "Reference: %{customdata[2]}"
        ])
    )
    
    fig.update_layout(
        width=1200,
        height=800,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    
    return fig

def ensure_unique_cluster_names(cluster_names, client):
    """Ensure cluster names are unique by regenerating duplicates"""
    seen_names = set()
    new_names = {}
    
    for cluster_id, name in cluster_names.items():
        if name in seen_names:
            # Generate new name with context
            prompt = f"""Le nom '{name}' est déjà utilisé pour un autre cluster. 
            Donne un nom plus spécifique à ce cluster. Retourne seulement le nouveau nom. Quelques mots maximum."""
            
            messages = [UserMessage(content=prompt)]
            
            try:
                response = client.chat.complete(
                    model="mistral-large-latest",
                    messages=messages,
                    temperature=0.7
                )
                new_name = response.choices[0].message.content.strip()
                # If still duplicate, add numerical suffix
                if new_name in seen_names:
                    new_name = f"{name} (Group {cluster_id + 1})"
                new_names[cluster_id] = new_name
                seen_names.add(new_name)
            except Exception:
                new_names[cluster_id] = f"{name} (Group {cluster_id + 1})"
                seen_names.add(new_names[cluster_id])
        else:
            new_names[cluster_id] = name
            seen_names.add(name)
    
    return new_names

def generate_trend_analysis(df, cluster_names, client):
    """Generate a human-readable analysis of the clusters using LLM"""
    # Prepare cluster information
    cluster_info = {}
    for cluster_id in df['Cluster'].unique():
        cluster_df = df[df['Cluster'] == cluster_id]
        # Get representative samples
        samples = cluster_df.sample(min(5, len(cluster_df)))
        
        cluster_info[cluster_id] = {
            'name': cluster_names[cluster_id],
            'size': len(cluster_df),
            'samples': [
                {'title': row['title'], 'body': row['body'][:200]} 
                for _, row in samples.iterrows()
            ]
        }
    
    # Create analysis prompt
    prompt = """Analysez ces groupes de propositions et rédigez un résumé éclairant pour les citoyens et les décideurs.
    À partir de ces données, concentrez-vous sur les thèmes principaux, les tendances et les implications potentielles pour les politiques publiques.
    Utilisez un langage clair et accessible en évitant les termes techniques.

    Structurez l'analyse comme suit :
    1. Tendances générales et thèmes principaux
    2. Observations notables pour chaque groupe
    3. Points clés à retenir pour les décideurs

    Groupes :
    """ 
    
    for cluster_id, info in cluster_info.items():
        prompt += f"\n{info['name']} ({info['size']} proposals)\n"
        prompt += "Sample proposals:\n"
        for sample in info['samples']:
            prompt += f"- {sample['title']}\n"
    
    messages = [UserMessage(content=prompt)]
    
    try:
        response = client.chat.complete(
            model="mistral-large-latest",
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating analysis: {str(e)}"

def suggest_new_categories(uncategorized_proposals, client):
    """Suggest new categories based on uncategorized proposals"""
    if len(uncategorized_proposals) == 0:
        return []
    
    # Sample proposals for analysis
    sample_size = min(10, len(uncategorized_proposals))
    samples = uncategorized_proposals.sample(n=sample_size)
    
    # Create analysis prompt
    prompt = """Analysez ces propositions non catégorisées et suggérez 3 à 5 nouvelles catégories distinctes.
    Les catégories doivent être :
    - Suffisamment spécifiques pour être pertinentes
    - Assez larges pour s'appliquer à plusieurs propositions
    - Clairement distinctes les unes des autres
    - Décrites en 2 à 4 mots chacune

    Ne retournez que les noms des catégories, séparés par des virgules.

    Propositions :
    """
    
    for _, row in samples.iterrows():
        prompt += f"Title: {row['title']}\n"
        prompt += f"Description: {row['body'][:200]}...\n\n"
    
    messages = [UserMessage(content=prompt)]
    
    try:
        response = client.chat.complete(
            model="mistral-large-latest",
            messages=messages,
            temperature=0.3
        )
        # Clean and validate suggested categories
        suggested_categories = [
            cat.strip() 
            for cat in response.choices[0].message.content.split(',')
            if 2 <= len(cat.strip().split()) <= 4  # Ensure proper length
        ]
        return suggested_categories[:5]  # Limit to 5 categories maximum
    except Exception as e:
        print(f"Error suggesting categories: {str(e)}")
        return []

def categorize_proposal_multi(proposal, categories, client):
    """Catégoriser une proposition dans plusieurs catégories"""
    prompt = f"""Voici les catégories disponibles :
    {', '.join(categories)}

    Pour cette proposition :
    Titre : {proposal['title']}
    Description : {proposal['body'][:500]}

    Attribuez TOUTES les catégories pertinentes. Gardez à l'esprit que :
    - Une proposition peut appartenir à plusieurs catégories si elles sont vraiment pertinentes
    - N'attribuez que les catégories qui correspondent fortement au contenu de la proposition
    - Si aucune catégorie ne convient bien, retournez "Non catégorisé"

    Ne retournez que les noms des catégories séparés par des virgules, rien d'autre."""

    messages = [UserMessage(content=prompt)]
    
    try:
        response = client.chat.complete(
            model="mistral-large-latest",
            messages=messages,
            temperature=0.3
        )
        assigned_categories = [
            cat.strip() 
            for cat in response.choices[0].message.content.split(',')
            if cat.strip() in categories
        ]
        return assigned_categories if assigned_categories else ["Uncategorized"]
    except Exception as e:
        print(f"Error categorizing proposal: {str(e)}")
        return ["Uncategorized"]

def create_interface():
    with gr.Blocks() as app:
        gr.Markdown("# IA data scientist pour Decidim")

        with gr.Tabs():
            # Exploration Tab
            with gr.Tab("Exploration"):
                with gr.Row():
                    with gr.Column():
                        base_url_explore = gr.Textbox(
                            label="Lien de la plateforme",
                            placeholder="par exemple https://www.lenomdelaplateforme.cat"
                        )
                        process_slug_explore = gr.Textbox(
                            label="Le process slug",
                            placeholder="e.g., nvelle-concertation"
                        )
                        n_clusters = gr.Slider(
                            minimum=2,
                            maximum=8,
                            value=4,
                            step=1,
                            label="Nombre du clusters"
                        )
                        analyze_btn = gr.Button("Analyser les propositions")

                with gr.Row():
                    cluster_plot = gr.Plot(label="Visualisation interactive")

                with gr.Row():
                    analysis_text = gr.Markdown(label="Analyse des tendences")

            # Categorization Tab
            with gr.Tab("Categorization"):
                with gr.Row():
                    with gr.Column():
                        base_url_cat = gr.Textbox(
                            label="Lien de la plateforme",
                            placeholder="e.g., https://develop.decidim-app.k8s.osp.cat/"
                        )
                        process_slug_cat = gr.Textbox(
                            label="Le process slug",
                            placeholder="e.g., nvlle-concertation"
                        )
                        allow_multiple = gr.Checkbox(
                            label="Permettre plusieurs catégories par proposition",
                            value=False
                        )
                        suggest_categories = gr.Checkbox(
                            label="Proposer de nouvelles catégories pour les propositions non catégorisées",
                            value=False
                        )
                        categories_input = gr.Textbox(
                            label="Les catégories (séparées par des virgules)",
                            placeholder="Par exemple: Transports, Parcs, Education, Logement",
                            lines=3
                        )
                        categorize_btn = gr.Button("Categorize Proposals")

                with gr.Row():
                    suggested_categories = gr.CheckboxGroup(
                        label="Changements suggérées",
                        choices=[],
                        visible=False
                    )

                with gr.Row():
                    results_table = gr.DataFrame(label="Résultats de la catégorisation")

                with gr.Row():
                    category_analysis = gr.Markdown(label="Analyse de la catégorisation")


        def exploration_handler(base_url, process_slug, n_clusters):
            try:
                client = MistralClient.get_instance()
                
                # Fetch and process data
                data = fetch_proposals(base_url, process_slug)
                df = process_proposals_data(data)

                # Generate embeddings and clustering
                df = generate_embeddings(df, client)
                df, matrix = perform_clustering(df, n_clusters)

                # Get cluster names and ensure uniqueness
                cluster_names, _ = analyze_clusters(df, client)
                cluster_names = ensure_unique_cluster_names(cluster_names, client)

                # Create visualization
                plot = create_interactive_cluster_viz(df, matrix, cluster_names)

                # Generate analysis
                analysis = generate_trend_analysis(df, cluster_names, client)

                return plot, analysis
            except EnvironmentError as e:
                return None, f"Erreur : {str(e)}"
            except Exception as e:
                return None, f"Une erreur est survenue : {str(e)}"

        def categorization_handler(base_url, process_slug, categories, allow_multiple, suggest_categories):
            try:
                client = MistralClient.get_instance()

                # Fetch and process data
                data = fetch_proposals(base_url, process_slug)
                df = process_proposals_data(data)

                # Convert categories string to list
                category_list = [cat.strip() for cat in categories.split(',')]

                # Initialize results
                results = []
                uncategorized = []

                # Process each proposal
                for _, row in df.iterrows():
                    if allow_multiple:
                        assigned_cats = categorize_proposal_multi(row, category_list, client)
                    else:
                        assigned_cats = categorize_proposal_multi(row, category_list, client)[:1]

                    if not assigned_cats and suggest_categories:
                        uncategorized.append(row)

                    results.append({
                        'id': row['id'],
                        'title': row['title'],
                        'categories': ', '.join(assigned_cats) if assigned_cats else 'Non catégorisé'
                    })

                results_df = pd.DataFrame(results)

                # Generate suggested categories if needed
                new_categories = []
                if suggest_categories and uncategorized:
                    new_categories = suggest_new_categories(pd.DataFrame(uncategorized), client)

                # Generate analysis
                analysis = generate_categorization_analysis(results_df, client)

                return (
                    results_df,
                    gr.CheckboxGroup(choices=new_categories, visible=bool(new_categories)),
                    analysis
                )
            except EnvironmentError as e:
                return pd.DataFrame(), gr.CheckboxGroup(visible=False), f"Erreur : {str(e)}"
            except Exception as e:
                return pd.DataFrame(), gr.CheckboxGroup(visible=False), f"Une erreur est survenue : {str(e)}"

        analyze_btn.click(
            exploration_handler,
            inputs=[base_url_explore, process_slug_explore, n_clusters],
            outputs=[cluster_plot, analysis_text]
        )

        categorize_btn.click(
            categorization_handler,
            inputs=[
                base_url_cat,
                process_slug_cat,
                categories_input,
                allow_multiple,
                suggest_categories
            ],
            outputs=[results_table, suggested_categories, category_analysis]
        )

    return app

if __name__ == "__main__":
    app = create_interface()
    app.launch()