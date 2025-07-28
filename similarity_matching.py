#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install sentence-transformers')


# In[16]:


import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# === Step 1: Load BIM and ICE data ===

bim_file = r'C:\Users\BaranMoradkhani\Desktop\project\BIM.csv'
ice_file = r'C:\Users\BaranMoradkhani\Desktop\project\ICE3.xlsx'

bim_df = pd.read_csv(bim_file)
ice_df = pd.read_excel(ice_file)

# === Step 2: Prepare text inputs for embedding ===

bim_df['text_for_embedding'] = 'query: ' + bim_df['Material Name'].fillna('')
ice_df['text_for_embedding'] = 'passage: ' + ice_df['Sawsan'].fillna('') + '; ' +'Sub-material: ' + ice_df['Sub-material'].fillna('')



# BIM side: combine Category, Name, Material Name, Grade, Density, ECI
#bim_df['text_for_embedding'] = (
    #'Category: ' + bim_df['Category'].fillna('') + '; ' +
    #'Name: ' + bim_df['Name'].fillna('') + '; ' +
    #'Material: ' + bim_df['Material Name'].fillna('') + '; ' +
    #'Grade: ' + bim_df['Grade'].fillna('') 
    #'Density: ' + bim_df['Density (kg/m3)'].astype(str).fillna('') + '; ' +
    #'ECI: ' + bim_df['ECI (kgCO2e/kg)'].astype(str).fillna('')
#)

# ICE side: combine Sawsan, Sub-material, ICE DB Name, Density, Carbon
#ice_df['text_for_embedding'] = (
    #'passage: ' +  # required for E5 model
    #'Material: ' + ice_df['Sawsan'].fillna('') + '; ' +
    #'Sub-material: ' + ice_df['Sub-material'].fillna('') + '; ' +
    #'ICE DB Name: ' + ice_df['ICE DB Name'].fillna('')
   # 'Density: ' + ice_df['Density of material - kg per m3'].astype(str).fillna('') + '; ' +
   # 'ECI: ' + ice_df['Embodied Carbon per kg (kg CO2e per kg)'].astype(str).fillna('')
#)


# === Step 3: Load pretrained model ===

model = SentenceTransformer('intfloat/e5-large-v2')

# === Step 4: Generate embeddings ===

print("Generating embeddings for BIM data...")
bim_embeddings = model.encode(bim_df['text_for_embedding'].tolist(), show_progress_bar=True)

print("Generating embeddings for ICE data...")
ice_embeddings = model.encode(ice_df['text_for_embedding'].tolist(), show_progress_bar=True)

# === Step 5: Calculate similarities and retrieve Top-3 matches ===

print("Computing similarity matrix...")
similarity_matrix = cosine_similarity(bim_embeddings, ice_embeddings)

top_n = 3
top_match_names = []
top_match_scores = []

for i in range(len(bim_df)):
    sims = similarity_matrix[i]
    top_indices = sims.argsort()[::-1][:top_n]

    top_names = [ice_df.iloc[idx]['Sawsan'] for idx in top_indices]
    top_scores = [sims[idx] for idx in top_indices]

    top_match_names.append(top_names)
    top_match_scores.append(top_scores)

    # Optional: Display for debugging
    print(f"\nBIM Material: {bim_df.iloc[i]['Material Name']}")
    for name, score in zip(top_names, top_scores):
        print(f"  → {name} (Score: {score:.3f})")

# === Step 6: Add results back to DataFrame ===

bim_df['Top 1 Match'] = [names[0] for names in top_match_names]
bim_df['Top 1 Score'] = [scores[0] for scores in top_match_scores]
bim_df['Top 2 Match'] = [names[1] if len(names) > 1 else None for names in top_match_names]
bim_df['Top 2 Score'] = [scores[1] if len(scores) > 1 else None for scores in top_match_scores]
bim_df['Top 3 Match'] = [names[2] if len(names) > 2 else None for names in top_match_names]
bim_df['Top 3 Score'] = [scores[2] if len(scores) > 2 else None for scores in top_match_scores]

#Confidence flag based on score
threshold = 0.7
bim_df['Match Confidence'] = np.where(bim_df['Top 1 Score'] >= threshold, 'High', 'Low')

# === Step 7: Save output ===

output_file = r'C:\Users\BaranMoradkhani\Desktop\project\OutPut_Top2.csv'
bim_df.to_csv(output_file, index=False)

print(f"\nMatching complete. Results saved to {output_file}")


# In[ ]:





# In[ ]:




