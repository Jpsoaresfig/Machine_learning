import os
import shutil
import pandas as pd

# Caminhos
csv_path = "train\_annotations_filtrado.csv"  # Ajusta aqui
images_dir = "train"  # Pasta onde estão as imagens
output_dir = "output"   # Pasta onde vai criar Rock/, Paper/, Scissors/

# Ler CSV
df = pd.read_csv(csv_path)

# Criar pastas de saída se não existirem
for classe in df['class'].unique():
    os.makedirs(os.path.join(output_dir, classe), exist_ok=True)

# Copiar imagens para a pasta correta
for idx, row in df.iterrows():
    filename = row['filename']
    classe = row['class']
    src_path = os.path.join(images_dir, filename)
    dst_path = os.path.join(output_dir, classe, filename)
    
    if os.path.isfile(src_path):
        shutil.copy2(src_path, dst_path)
    else:
        print(f"Arquivo não encontrado: {src_path}")
