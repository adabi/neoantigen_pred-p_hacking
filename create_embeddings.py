import esm
import torch
import pandas as pd
from tqdm import tqdm

# Load anitgen xlsx file in ./data and combine all sheets into one dataframe
df_neoantingens = pd.read_excel('./data/nature24473_MOESM4_neoantigens.xlsx', sheet_name=None)
#df = pd.read_excel(input_file , sheet_name=None)
df_neoantingens = pd.concat(df_neoantingens.values(), ignore_index=True)


#Load the preprocessed patients csv file
df_survival = pd.read_csv('./data/processed/patients.csv')

#left join the two dataframes with 'Sample' as key and 0 as default value
df = pd.merge(df_neoantingens, df_survival, on='Sample', how='left').fillna(0)

# select only the columns we need
df = df[['Sample', 'MUTATION_ID', 'MT.Peptide', 'year_death', 'Status', 'Months']]

#create list of tuples with (MUTATION_ID, MT.Peptide) from the dataframe
data = [(row['MUTATION_ID'], row['MT.Peptide']) for index, row in df.iterrows()]

#split the data into batches 
batch_size = 512
data = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]


# Using the general purpose ESM-2 model
model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
batch_converter = alphabet.get_batch_converter()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model.to(device)

model.eval()  # disables dropout for deterministic result

sequence_representations = []

for batch in tqdm(data):
    batch_labels, batch_strs, batch_tokens = batch_converter(batch)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)

    token_representations = results["representations"][33]

    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))

# convert to numpy array
sequence_representations = torch.stack(sequence_representations).cpu().numpy()

# add sequence representations to dataframe, name the columns as 'embedding_0', 'embedding_1', etc.
embedding_columns = [f'embedding_{i}' for i in range(sequence_representations.shape[1])]
embedding_df = pd.DataFrame(sequence_representations, columns=embedding_columns)
df = pd.concat([df, embedding_df], axis=1)

# drop the 'MT.Peptide' and 'MUTATION_ID column
df = df.drop(columns=['MT.Peptide', 'MUTATION_ID'])

# replace that one dash in the Status column with 0
df['Status'] = df['Status'].replace('-', 0)

#convert Status columnt to int
df['Status'] = df['Status'].astype(int)

#save dataframe to feather file
df.to_feather('./data/processed/embeddings.feather')



